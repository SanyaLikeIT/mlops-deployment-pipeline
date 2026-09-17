# Wine MLOps Pipeline

A small end-to-end MLOps project for classifying wine cultivars. The project covers the complete assignment flow: raw data preparation, model training and evaluation, experiment tracking, model serving through an API, a browser-based prediction app, Docker deployment, and scheduled execution every five minutes.

## Dataset

The raw data comes from scikit-learn's bundled Wine dataset:

```python
from sklearn.datasets import load_wine
load_wine(as_frame=True)
```

The dataset contains 178 samples, 13 numeric source features, and 3 cultivar classes (`class_0`, `class_1`, and `class_2`). `code/datasets/create_raw_data.py` materializes it as `data/raw/wine.csv`, so after the file is created the normal pipeline does not need to download data from the internet.

The deployed model intentionally uses six features so the web form stays simple:

- `alcohol`
- `malic_acid`
- `magnesium`
- `flavanoids`
- `color_intensity`
- `proline`

The processed CSV files still keep all 13 source features.

## Pipeline architecture

```text
sklearn.datasets.load_wine()
          |
          v
 data/raw/wine.csv
          |
          v
 DVC: prepare_data
 - schema validation
 - duplicate removal
 - median imputation
 - IQR outlier removal
 - stratified 80/20 split
          |
          +-------------------+
          |                   |
          v                   v
 train.csv                 test.csv
          \                   /
           \                 /
            v               v
              DVC: train_model
       six selected model features
       StandardScaler
       LogisticRegression
       accuracy + macro F1
       MLflow experiment logging
                    |
                    v
          models/wine_model.joblib
                    |
                    v
              FastAPI service
                    ^
                    | HTTP
                    |
              Streamlit app
```

The application is more than a standalone ML script because the data and training steps are reproducible with DVC, experiments are recorded with MLflow, the fitted preprocessing/model pipeline is packaged, predictions are exposed through an HTTP API, the frontend and API run in separate containers, and cron can rerun the complete pipeline automatically.

## Tools used

- **DVC** describes dependencies between data-processing and training stages and reproduces them in the correct order.
- **MLflow** records training parameters, metrics, and artifacts for each model run.
- **scikit-learn** provides preprocessing and the classifier.
- **FastAPI** exposes the trained model through `/predict` and provides `/health` for service checks.
- **Streamlit** provides the browser form used to enter measurements and display predictions.
- **Docker** packages the API and app with their own dependencies.
- **Docker Compose** starts the two containers together and provides networking between them.
- **cron** invokes the full pipeline every five minutes.

## Repository structure

```text
.
├── code
│   ├── datasets
│   │   ├── create_raw_data.py
│   │   └── prepare_data.py
│   ├── models
│   │   └── train.py
│   └── deployment
│       ├── api
│       │   ├── main.py
│       │   ├── requirements.txt
│       │   └── Dockerfile
│       ├── app
│       │   ├── app.py
│       │   ├── requirements.txt
│       │   └── Dockerfile
│       └── docker-compose.yml
├── data
│   ├── raw
│   │   └── wine.csv
│   └── processed
├── models
├── scripts
│   ├── run_pipeline.sh
│   ├── install_cron.sh
│   └── smoke_test.py
├── dvc.yaml
├── dvc.lock
├── requirements.txt
└── README.md
```

`data/processed/`, model binaries, MLflow runs, caches, and logs are generated locally and are excluded from normal Git tracking where appropriate.

## Ubuntu setup

### 1. Create a virtual environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Verify Docker

Docker and the Compose plugin must be installed before the deployment stage:

```bash
docker --version
docker compose version
docker run hello-world
```

If Docker is not installed, install Docker Engine and the Docker Compose plugin using the official Docker instructions for your Ubuntu release.

If Docker works only with `sudo`, configure your user for Docker access and then log out and back in before using the automated pipeline. The scheduled job cannot interactively ask for a sudo password.

## Running the project

### Generate the raw dataset

The CSV is already included in the repository, but it can be recreated if it is missing:

```bash
python code/datasets/create_raw_data.py
```

### Run the data and model stages with DVC

```bash
dvc repro -f
```

This runs:

1. `prepare_data`: raw CSV -> cleaned train/test CSV files.
2. `train_model`: train/test CSV files -> metrics and packaged model.

The current test metrics are stored in:

```text
models/metrics.json
```

DVC also keeps stage dependency/output hashes in `dvc.lock` for reproducibility.

### Run the complete pipeline

```bash
./scripts/run_pipeline.sh
```

The script:

1. activates `.venv` when it exists;
2. prevents overlapping runs with `flock`;
3. creates the raw dataset if needed;
4. executes `dvc repro -f`;
5. builds/recreates the FastAPI and Streamlit containers;
6. waits for healthy services before finishing.

After a successful run:

- Streamlit app: http://localhost:8501
- FastAPI documentation: http://localhost:8000/docs
- FastAPI health endpoint: http://localhost:8000/health

## Prediction API

The API expects the same six feature names used during model training.

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "alcohol": 13.2,
    "malic_acid": 1.8,
    "magnesium": 100.0,
    "flavanoids": 2.5,
    "color_intensity": 5.0,
    "proline": 900.0
  }'
```

The response contains the predicted class and probabilities for all three classes, for example:

```json
{
  "prediction": "class_0",
  "probabilities": {
    "class_0": 0.91,
    "class_1": 0.08,
    "class_2": 0.01
  }
}
```

The exact probabilities depend on the fitted model.

## MLflow experiment tracking

Training uses a local MLflow file store under `mlruns/` and the experiment name `wine-classification`.

Start the MLflow UI from the repository root:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open:

```text
http://localhost:5000
```

Each run records the selected features, model/scaler information, train/test row counts, accuracy, macro F1, and model/metric artifacts.

## Smoke test

With the Docker services running:

```bash
python scripts/smoke_test.py
```

It checks `/health`, submits a prediction request, validates the returned class, and verifies that class probabilities are valid and sum approximately to 1.

## Automatic execution every five minutes

Install the repository-specific cron entry once:

```bash
./scripts/install_cron.sh
```

Inspect the installed schedule:

```bash
crontab -l
```

The schedule is:

```cron
*/5 * * * *
```

Pipeline stdout/stderr is appended to:

```text
pipeline.log
```

Follow it with:

```bash
tail -f pipeline.log
```

The installer is idempotent: running it again does not add another identical entry and does not remove unrelated cron jobs.

## Stopping the services

```bash
docker compose -f code/deployment/docker-compose.yml down
```

## Troubleshooting

### Docker permission denied

Check:

```bash
docker info
```

If access is denied, configure non-root Docker access for your Ubuntu user, then log out and back in.

### Docker daemon is not running

Check:

```bash
sudo systemctl status docker
```

Start it if necessary:

```bash
sudo systemctl start docker
```

### Port 8000 or 8501 is already in use

Find the process/container using the port or stop the conflicting service before starting this Compose project.

### Model file is missing

Run:

```bash
dvc repro -f
```

Then verify:

```text
models/wine_model.joblib
```

and start the deployment again.

### cron cannot find Python/DVC

Create the project `.venv`, install `requirements.txt`, and verify that `./scripts/run_pipeline.sh` works manually first. The script activates `.venv` automatically for scheduled runs.

## Assignment requirements mapping

| Requirement | Implementation |
| --- | --- |
| Data Engineering | `data/raw/wine.csv`, `code/datasets/prepare_data.py`, DVC `prepare_data` stage |
| Data loading | `pd.read_csv()` in `prepare_data.py` |
| Missing-value handling | invalid numeric values -> NaN -> feature median imputation |
| Outlier removal | 1.5×IQR rule over numeric Wine features |
| Train/test artifacts | `data/processed/train.csv`, `data/processed/test.csv` |
| Feature engineering | six-feature selection + `StandardScaler` |
| Model training | `LogisticRegression` in `code/models/train.py` |
| Evaluation | accuracy and macro F1 in `models/metrics.json` |
| Experiment tracking | local MLflow experiment `wine-classification` |
| Model packaging | complete fitted pipeline in `models/wine_model.joblib` |
| Model API | FastAPI in `code/deployment/api/main.py` |
| Web application | Streamlit in `code/deployment/app/app.py` |
| Separate containers | API and app services in `code/deployment/docker-compose.yml` |
| App -> API communication | HTTP POST from Streamlit to `http://api:8000/predict` inside the Compose network |
| Full pipeline automation | `scripts/run_pipeline.sh` |
| Five-minute schedule | `scripts/install_cron.sh` |
| Repository structure | separated `code/`, `data/`, `models/`, and `scripts/` directories |
