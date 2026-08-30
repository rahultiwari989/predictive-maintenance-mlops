# Predictive Maintenance MLOps

End-to-end Remaining Useful Life (RUL) prediction based on the uploaded `Preventive_Maintenance.ipynb`.

## Current phase

Phase 1 converts the notebook into a reproducible Python training/inference project.

- C-MAPSS FD001 input
- 8 non-informative columns removed, matching the notebook
- RUL target with cap = 125
- 30-cycle sliding windows
- 16 model features
- engine-exclusive train/validation split
- StandardScaler persisted as an artifact
- TensorFlow CNN-LSTM baseline structure preserved from the notebook
- FastAPI inference service with health/readiness endpoints

## Run training

Place `train_FD001.txt` in `data/`, then:

```bash
python -m src.train
```

This creates model/scaler/metadata artifacts.

## Run API locally

```bash
uvicorn api.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for Swagger UI.

## Important model note

The notebook creates a Conv1D + MaxPooling branch but then passes the original `inputs` directly into the first LSTM. The Phase 1 implementation intentionally preserves that behavior so we can establish a reproducible baseline before changing the architecture.

## Next phases

1. Validate training and inference locally.
2. Add tests.
3. Add Streamlit as a client of FastAPI.
4. Docker Compose.
5. Kubernetes Deployment/Service/probes/HPA.
6. Sensor simulator for mock production telemetry.
7. GitHub Actions CI/CD.
