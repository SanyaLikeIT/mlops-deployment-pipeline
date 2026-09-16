"""Train a six-feature Wine pipeline and record a local MLflow experiment."""
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ["alcohol", "malic_acid", "magnesium", "flavanoids", "color_intensity", "proline"]


def read_split(name):
    frame = pd.read_csv(ROOT / f"data/processed/{name}.csv")
    missing = set(FEATURES + ["target"]) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} CSV is missing required columns: {sorted(missing)}")
    if frame.empty or not np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError(f"{name} CSV is empty or has invalid numeric values; run preparation first")
    if not frame["target"].isin({"class_0", "class_1", "class_2"}).all():
        raise ValueError(f"{name} CSV contains missing or unknown target labels")
    return frame


def main():
    train, test = read_split("train"), read_split("test")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("wine-classification")
    with mlflow.start_run(run_name="logistic-regression"):
        mlflow.log_params({
            "model_type": "LogisticRegression", "scaler": "StandardScaler",
            "selected_features": ",".join(FEATURES), "feature_count": len(FEATURES),
            "max_iter": 1000, "random_state": 42, "test_size": 0.2,
            "train_rows": len(train), "test_rows": len(test),
        })
        model.fit(train[FEATURES], train["target"])
        predictions = model.predict(test[FEATURES])
        metrics = {
            "accuracy": float(accuracy_score(test["target"], predictions)),
            "macro_f1": float(f1_score(test["target"], predictions, average="macro")),
        }
        output = ROOT / "models"
        output.mkdir(parents=True, exist_ok=True)
        # Atomic replacement prevents readers from opening a partial model file.
        temporary = output / "wine_model.joblib.tmp"
        joblib.dump(model, temporary)
        temporary.replace(output / "wine_model.joblib")
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(output / "metrics.json"))
        mlflow.log_artifact(str(output / "wine_model.joblib"))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Model: {output / 'wine_model.joblib'}; experiment history: {ROOT / 'mlruns'}")


if __name__ == "__main__":
    main()
