import math
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flows: list[dict[str, float | None]] = Field(
        min_length=1,
        max_length=100,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    default_path = (
        Path(__file__).resolve().parent / "artifacts/model.joblib"
    )
    model_path = Path(os.getenv("MODEL_PATH", str(default_path)))

    # Only load model files you trust.
    bundle = joblib.load(model_path)

    app.state.pipeline = bundle["pipeline"]
    app.state.features = bundle["features"]
    app.state.model_name = bundle["model_name"]
    yield


app = FastAPI(
    title="AI Intrusion Detection API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": app.state.model_name,
    }


@app.get("/schema")
def schema():
    return {
        "features": app.state.features,
        "labels": ["benign", "attack"],
        "threshold": 0.5,
    }


@app.post("/predict")
def predict(payload: PredictionRequest):
    expected = set(app.state.features)

    for index, flow in enumerate(payload.flows):
        provided = set(flow)

        if provided != expected:
            raise HTTPException(
                status_code=422,
                detail={
                    "row": index,
                    "missing": sorted(expected - provided),
                    "unexpected": sorted(provided - expected),
                },
            )

        if all(value is None for value in flow.values()):
            raise HTTPException(
                status_code=422,
                detail="A flow cannot contain only null values.",
            )

        for value in flow.values():
            if value is not None and (
                not math.isfinite(value)
                or abs(value) > 3.4028234663852886e38
            ):
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid numeric value in row {index}.",
                )

    frame = pd.DataFrame(
        payload.flows,
        columns=app.state.features,
        dtype=float,
    )

    scores = app.state.pipeline.predict_proba(frame)[:, 1]

    return {
        "model": app.state.model_name,
        "predictions": [
            {
                "label": "attack" if score >= 0.5 else "benign",
                "attack_score": float(score),
            }
            for score in scores
        ],
    }
