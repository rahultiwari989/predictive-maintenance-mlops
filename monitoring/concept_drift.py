"""Delayed-label performance monitoring; inputs alone cannot prove concept drift."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_labeled_predictions(frame: pd.DataFrame) -> dict:
    required = {"predicted_rul", "actual_rul"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    clean = frame[["predicted_rul", "actual_rul"]].apply(pd.to_numeric, errors="coerce").dropna()
    if clean.empty:
        raise ValueError("No valid labeled prediction rows")
    y_true, y_pred = clean["actual_rul"], clean["predicted_rul"]
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    threshold = float(os.environ.get("MAX_RUL_RMSE", "20"))
    return {"rows": int(len(clean)), "mae": mae, "rmse": rmse,
            "max_allowed_rmse": threshold,
            "status": "degraded" if rmse > threshold else "ok",
            "note": "Threshold is a configurable starting point; calibrate on operational validation data."}


def main() -> int:
    labels_path = Path(os.environ.get("LABELED_PREDICTIONS", "monitoring/labeled_predictions.csv"))
    output_path = Path(os.environ.get("PERFORMANCE_REPORT", "monitoring/reports/model_performance.json"))
    if not labels_path.exists():
        print(f"Skipping performance check: labeled outcomes not found at {labels_path}")
        return 0
    report = evaluate_labeled_predictions(pd.read_csv(labels_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
