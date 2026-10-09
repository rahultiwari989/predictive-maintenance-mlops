import json
import logging
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import RULPredictor

app = FastAPI(title="Predictive Maintenance RUL API", version="0.2.0")
logger = logging.getLogger("rul_monitoring")
predictor = None


class PredictionRequest(BaseModel):
    sensor_window: list[list[float]] = Field(..., description="30 x 16 feature matrix")


@app.on_event("startup")
def load_model():
    global predictor
    predictor = RULPredictor()


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": predictor is not None}


@app.get("/ready")
def ready():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        rul = predictor.predict(request.sensor_window)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # JSON-structured stdout logs are collected by the GKE logging agent.
    # Do not log the full time-series window; log per-feature latest values
    # and rolling mean/std so downstream jobs can monitor distributions.
    window = np.asarray(request.sensor_window, dtype=float)
    feature_names = getattr(predictor, "feature_cols", None) or getattr(
        predictor, "metadata", {}
    ).get("feature_cols", [])
    stats = {}
    for idx in range(window.shape[1]):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        stats[name] = {
            "latest": float(window[-1, idx]),
            "mean_window": float(np.mean(window[:, idx])),
            "std_window": float(np.std(window[:, idx])),
        }
    event = {
        "event": "rul_prediction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "predicted_rul": float(rul),
        "unit": "cycles",
        "status": "critical" if rul <= 10 else "warning" if rul <= 30 else "healthy",
        "model_version": getattr(predictor, "metadata", {}).get("model_version", "unknown"),
        "feature_stats": stats,
        "label_available": False,
    }
    logger.info(json.dumps(event, separators=(",", ":")))
    return {
        "predicted_rul": rul,
        "unit": "cycles",
        "status": event["status"],
    }
