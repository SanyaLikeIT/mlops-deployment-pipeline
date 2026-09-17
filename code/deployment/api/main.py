"""Serve predictions from the fitted Wine classification pipeline."""
from contextlib import asynccontextmanager
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

FEATURES = ["alcohol", "malic_acid", "magnesium", "flavanoids", "color_intensity", "proline"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.environ.get("MODEL_PATH", "/models/wine_model.joblib"))
    if not path.is_file():
        raise RuntimeError(f"Model missing: {path}. Run dvc repro -f before starting the API.")
    app.state.model = joblib.load(path)
    yield


app = FastAPI(title="Wine Prediction API", lifespan=lifespan)


class WineMeasurements(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alcohol: float = Field(gt=0, allow_inf_nan=False)
    malic_acid: float = Field(gt=0, allow_inf_nan=False)
    magnesium: float = Field(gt=0, allow_inf_nan=False)
    flavanoids: float = Field(ge=0, allow_inf_nan=False)
    color_intensity: float = Field(gt=0, allow_inf_nan=False)
    proline: float = Field(gt=0, allow_inf_nan=False)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(measurements: WineMeasurements):
    frame = pd.DataFrame([measurements.model_dump()], columns=FEATURES)
    model = app.state.model
    probabilities = model.predict_proba(frame)[0]
    return {
        "prediction": str(model.predict(frame)[0]),
        "probabilities": dict(zip(model.classes_.tolist(), probabilities.tolist())),
    }
