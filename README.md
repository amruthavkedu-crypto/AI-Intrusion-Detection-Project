# AI-Intrusion-Detection-Project
# AI-Powered Intrusion Detection System

A machine-learning project that classifies network flows as **benign or attack**, using Random Forest and XGBoost. The selected model is served through a FastAPI REST API packaged with Docker.

## Technology Stack

- Python, Pandas, NumPy, scikit-learn
- Random Forest and XGBoost
- FastAPI and Uvicorn
- Docker
- Google Colab for training
- GitHub Codespaces for container testing

## Workflow

1. Load and sample CICIDS2017 network-flow CSVs.
2. Remove identifiers and duplicate or conflicting feature rows.
3. Split the data into training and test sets.
4. Tune both classifiers using three-fold cross-validation.
5. Select the model using training cross-validation average precision.
6. Evaluate both models on the held-out test set.
7. Serve the selected model through FastAPI and Docker.

Missing-value imputation is fitted inside each cross-validation training fold.

## Dataset

This project uses the machine-learning CSV files from [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html), collected in a controlled network environment.

The experiment sampled up to **20,000 rows per CSV**. After duplicate and conflicting-row removal:

- Cleaned sample: **154,122 rows**
- Training set: **123,297 rows**
- Test set: **30,825 rows**
- Split: stratified random 80/20
- Random seed: **42**

Labels are converted to binary classes: `BENIGN` and `attack`.

## Results

| Model | Attack precision | Attack recall | Attack F1 | False-positive rate | Average precision |
|---|---:|---:|---:|---:|---:|
| Random Forest | 99.27% | 99.37% | 99.32% | 0.17% | 0.9994 |
| XGBoost | 99.57% | 99.74% | 99.65% | 0.10% | 0.9998 |

**Selected model: XGBoost**, chosen using training cross-validation scores.

These are rounded results from the sampled, randomly split dataset. They do not establish performance on unseen networks. Full results are available in `reports/metrics.json` and `reports/comparison.csv`.

## Project Structure

```text
api.py                 FastAPI application
train.py               Data preparation, tuning, and evaluation
requirements.txt       Python dependencies
Dockerfile             Container build instructions
.dockerignore          Docker build exclusions
.gitignore             Git exclusions
reports/               Measured evaluation results
```

Training generates an `artifacts/` folder containing the model and example request. Model files and raw datasets are excluded from Git.

## Train the Models

Download and extract the CICIDS2017 machine-learning CSVs into a `data/` folder.

Use the Python major/minor version specified in the Dockerfile.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py --data-dir data --rows-per-file 20000
```

The activation command above is for Linux/macOS.

Training generates:

```text
artifacts/model.joblib
artifacts/example_request.json
reports/metrics.json
```

`reports/comparison.csv` was generated separately in the Colab results cell.

## Run with Docker

Train first, or restore your own trusted `model.joblib` and `example_request.json` into `artifacts/`.

```bash
docker build -t ai-ids:1.0 .
docker run --rm -p 8000:8000 ai-ids:1.0
```

Open `http://localhost:8000/docs` locally. In Codespaces, open the forwarded port 8000 URL and append `/docs`.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check service status and selected model |
| GET | `/schema` | Retrieve required feature names |
| POST | `/predict` | Classify a batch of network flows |

Test a prediction from another terminal:

```bash
curl --fail-with-body -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data-binary "@artifacts/example_request.json"
```

Requests must use the same feature names, definitions, and units used during training. The API accepts up to 100 flows per request.

## Validation Completed

- Trained and tuned both classifiers.
- Evaluated both models on a held-out test set.
- Verified health and prediction requests using FastAPI’s test client.
- Built and started the Docker container.
- Verified prediction requests from the terminal.
- Opened the interactive API documentation through Codespaces.

## Limitations

- Binary classification; individual attack types are not distinguished.
- No live packet capture, flow extraction, or automatic blocking.
- Random splitting can leave related flows across training and test sets.
- Per-file sampling changes the original dataset proportions.
- Attack scores are not calibrated confidence estimates.
- Latency, throughput, and scalability have not been benchmarked.
- Codespaces is used for development and demonstration, not permanent hosting.

## Future Improvements

- Evaluate on separate capture days and external datasets.
- Select the decision threshold using validation data.
- Add multiclass attack classification.
- Integrate compatible live flow extraction.
- Add authentication, load testing, and drift monitoring.

## Dataset Citation

Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani.  
“Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.” ICISSP, 2018.
