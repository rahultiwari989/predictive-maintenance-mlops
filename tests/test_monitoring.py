import pandas as pd

from monitoring.data_drift import compare_frames, psi
from monitoring.concept_drift import evaluate_labeled_predictions
from monitoring.retraining_gate import approve_candidate


def test_psi_is_near_zero_for_identical_samples():
    values = pd.Series(range(100))
    assert psi(values, values) < 1e-6


def test_drift_report_includes_feature_statistics():
    reference = pd.DataFrame({"sensor_2": range(100)})
    current = pd.DataFrame({"sensor_2": range(100, 200)})
    report = compare_frames(reference, current, ["sensor_2"])
    assert "psi" in report["sensor_2"]
    assert report["sensor_2"]["status"] == "drift"


def test_labeled_performance_reports_rmse_and_mae():
    frame = pd.DataFrame({"predicted_rul": [10, 20], "actual_rul": [12, 18]})
    report = evaluate_labeled_predictions(frame)
    assert report["mae"] == 2.0
    assert report["rmse"] == 2.0


def test_gate_rejects_missing_smoke_test_evidence():
    candidate = {"validation_rmse": 10, "schema_valid": True}
    champion = {"validation_rmse": 11}
    approved, reasons = approve_candidate(candidate, champion)
    assert not approved
    assert any("smoke-test" in reason for reason in reasons)


def test_gate_approves_only_candidate_with_evidence_and_better_metric():
    candidate = {"validation_rmse": 10, "schema_valid": True, "smoke_test_passed": True}
    champion = {"validation_rmse": 11}
    approved, _ = approve_candidate(candidate, champion)
    assert approved
