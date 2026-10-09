"""Feature drift checks using PSI and two-sample KS tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index using reference quantile bins."""
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if len(ref) < 2 or len(cur) < 2:
        raise ValueError("Reference and current samples must each contain at least 2 values")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=edges)[0] / len(cur)
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compare_frames(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> dict:
    """Return per-feature drift statistics; PSI bands are initial heuristics."""
    report = {}
    for feature in features:
        if feature not in reference or feature not in current:
            report[feature] = {"status": "missing_feature"}
            continue
        ref = pd.to_numeric(reference[feature], errors="coerce").dropna()
        cur = pd.to_numeric(current[feature], errors="coerce").dropna()
        if len(ref) < 2 or len(cur) < 2:
            report[feature] = {"status": "insufficient_data"}
            continue
        stat, p_value = ks_2samp(ref, cur)
        score = psi(ref, cur)
        band = "low" if score < 0.1 else "investigate" if score < 0.25 else "high"
        report[feature] = {
            "psi": score, "psi_band": band,
            "ks_statistic": float(stat), "ks_p_value": float(p_value),
            "reference_count": int(len(ref)), "current_count": int(len(cur)),
            "status": "drift" if score >= 0.25 or p_value < 0.01 else "ok",
        }
    return report


def main() -> int:
    reference_path = Path(__import__("os").environ.get("REFERENCE_DATA", "monitoring/reference.csv"))
    current_path = Path(__import__("os").environ.get("CURRENT_DATA", "monitoring/current.csv"))
    output_path = Path(__import__("os").environ.get("DRIFT_REPORT", "monitoring/reports/data_drift.json"))
    if not reference_path.exists() or not current_path.exists():
        print(f"Skipping drift check: expected {reference_path} and {current_path}")
        return 0
    reference, current = pd.read_csv(reference_path), pd.read_csv(current_path)
    features = [c for c in reference.columns if c in current.columns]
    report = compare_frames(reference, current, features)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    high = [k for k, v in report.items() if v.get("status") == "drift"]
    print(json.dumps({"report": str(output_path), "drifted_features": high}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
