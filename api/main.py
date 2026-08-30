from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import RULPredictor

app = FastAPI(title="Predictive Maintenance RUL API", version="0.1.0")
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
    return {
        "predicted_rul": rul,
        "unit": "cycles",
        "status": "critical" if rul <= 10 else "warning" if rul <= 30 else "healthy",
    }
