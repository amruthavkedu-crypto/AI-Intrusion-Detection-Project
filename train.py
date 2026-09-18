import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

SEED = 42

EXCLUDED = {
    "label", "flow id", "source ip", "destination ip",
    "src ip", "dst ip", "timestamp",
}


def load_data(data_dir, rows_per_file):
    files = sorted(
        path for path in Path(data_dir).rglob("*")
        if path.is_file() and path.suffix.lower() == ".csv"
    )

    if not files:
        raise ValueError("No CSV files found in the dataset folder.")

    parts = []
    expected_columns = None
    manifest = []

    for file in files:
        print(f"Reading {file.name}", flush=True)

        frame = pd.read_csv(file, low_memory=False)
        frame.columns = frame.columns.str.strip()

        if frame.columns.duplicated().any():
            raise ValueError(f"Duplicate column names in {file.name}")

        label_columns = [
            column for column in frame.columns
            if column.casefold() == "label"
        ]

        if len(label_columns) != 1:
            raise ValueError(f"Expected one Label column in {file.name}")

        original_rows = len(frame)

        frame = frame.sample(
            n=min(rows_per_file, len(frame)),
            random_state=SEED,
        ).copy()

        labels = frame[label_columns[0]].astype("string").str.strip()
        valid = labels.notna() & labels.ne("")
        frame = frame.loc[valid].copy()
        labels = labels.loc[valid]

        feature_columns = [
            column for column in frame.columns
            if column.casefold() not in EXCLUDED
            and not column.casefold().startswith("unnamed:")
        ]

        if expected_columns is None:
            expected_columns = feature_columns
        elif set(feature_columns) != set(expected_columns):
            raise ValueError(
                f"Features differ in {file.name}. "
                "Use CSV files from the same dataset archive."
            )

        features = frame[expected_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.mask(
            features.abs() > np.finfo(np.float32).max
        )
        features = features.astype("float32")

        features["target"] = (
            labels.str.upper() != "BENIGN"
        ).astype("int8")

        parts.append(features)

        manifest.append({
            "file": file.name,
            "original_rows": original_rows,
            "sampled_labeled_rows": len(features),
        })

    data = pd.concat(parts, ignore_index=True)

    # Remove duplicate feature rows and rows with conflicting labels.
    hashes = pd.util.hash_pandas_object(
        data[expected_columns], index=False
    )
    conflicts = (
        data["target"].groupby(hashes).transform("nunique") > 1
    )
    keep = ~conflicts & ~hashes.duplicated()

    print(
        f"Removed {int((~keep).sum())} duplicate/conflicting rows.",
        flush=True,
    )

    data = data.loc[keep].reset_index(drop=True)
    return data, expected_columns, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--rows-per-file", type=int, default=20000)
    args = parser.parse_args()

    if args.rows_per_file < 1:
        raise ValueError("--rows-per-file must be positive.")

    # Save outputs beside this script, regardless of the working folder.
    project = Path(__file__).resolve().parent
    artifacts = project / "artifacts"
    reports = project / "reports"
    artifacts.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)

    data, columns, manifest = load_data(
        args.data_dir, args.rows_per_file
    )

    counts = data["target"].value_counts()
    if len(counts) != 2 or counts.min() < 20:
        raise ValueError(
            "Need at least 20 benign and 20 attack rows. "
            "Include CSV files containing attack traffic."
        )

    print(f"Total cleaned sample: {len(data):,} rows", flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        data[columns],
        data["target"],
        test_size=0.2,
        stratify=data["target"],
        random_state=SEED,
    )

    # Select usable columns using training data only.
    columns = X_train.columns[X_train.notna().any()].tolist()
    if not columns:
        raise ValueError("No usable numeric features found.")

    X_train = X_train[columns]
    X_test = X_test[columns]

    candidates = {
        "random_forest": (
            RandomForestClassifier(
                n_estimators=150,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=2,
            ),
            {"model__max_depth": [12, None]},
        ),
        "xgboost": (
            XGBClassifier(
                n_estimators=150,
                learning_rate=0.1,
                tree_method="hist",
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=2,
            ),
            {"model__max_depth": [4, 8]},
        ),
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED,
    )
    searches = {}

    for name, (estimator, parameters) in candidates.items():
        print(f"\nTuning {name}...", flush=True)

        pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            ("model", estimator),
        ])

        search = GridSearchCV(
            pipeline,
            parameters,
            scoring="average_precision",
            cv=cv,
            n_jobs=1,
            error_score="raise",
            verbose=1,
        )

        search.fit(X_train, y_train)
        searches[name] = search

    # Choose the model using training cross-validation, not test scores.
    winner = max(
        searches,
        key=lambda name: searches[name].best_score_,
    )

    results = {}

    for name, search in searches.items():
        model = search.best_estimator_
        predictions = model.predict(X_test)
        scores = model.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        results[name] = {
            "best_parameters": search.best_params_,
            "cv_average_precision": float(search.best_score_),
            "test_average_precision": float(
                average_precision_score(y_test, scores)
            ),
            "test_roc_auc": float(roc_auc_score(y_test, scores)),
            "false_positive_rate": float(fp / (fp + tn)),
            "confusion_matrix": [
                [int(tn), int(fp)],
                [int(fn), int(tp)],
            ],
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["benign", "attack"],
                output_dict=True,
                zero_division=0,
            ),
        }

    joblib.dump(
        {
            "pipeline": searches[winner].best_estimator_,
            "features": columns,
            "model_name": winner,
        },
        artifacts / "model.joblib",
    )

    metadata = {
        "selected_model": winner,
        "seed": SEED,
        "sampling": f"Up to {args.rows_per_file} rows per CSV",
        "split": "Stratified random 80/20 after deduplication",
        "selection": "Training CV average precision",
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_attack_fraction": float(y_test.mean()),
        "input_files": manifest,
        "results": results,
    }

    (reports / "metrics.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    sample = {
        name: None if pd.isna(value) else float(value)
        for name, value in X_test.iloc[0].items()
    }

    (artifacts / "example_request.json").write_text(
        json.dumps(
            {"flows": [sample]},
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print(f"\nSelected model: {winner}", flush=True)
    print("Saved model, metrics, and example request.", flush=True)


if __name__ == "__main__":
    main()
