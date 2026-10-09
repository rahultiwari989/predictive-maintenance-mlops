# Airflow orchestration

These DAGs are an orchestration layer, not a bundled Airflow deployment. Install Airflow in a separate environment/Composer and mount or package this repository at `PROJECT_DIR` (default `/opt/predictive-maintenance`).

## DAGs

- `predictive_maintenance_model_monitoring`: every six hours, runs feature-drift checks and delayed-label performance evaluation.
- `predictive_maintenance_retraining_gate`: manual DAG; fails closed unless candidate/champion metadata exist and the candidate passes the configured RMSE/schema/smoke-test checks.

## Monitoring data contract

Configure environment variables for paths:
- `REFERENCE_DATA`: baseline feature CSV created from training data.
- `CURRENT_DATA`: recent production feature sample CSV.
- `LABELED_PREDICTIONS`: CSV with `predicted_rul,actual_rul`; populate only when verified labels arrive.
- `CANDIDATE_REPORT`: candidate JSON containing `validation_rmse` (and ideally `schema_valid`, `smoke_test_passed`).
- `CHAMPION_REPORT`: current champion metadata JSON with `validation_rmse`.

The monitor skips gracefully if optional monitoring inputs do not exist. Configure durable storage and a producer for these CSVs before treating reports as operational alerts. Do not infer concept drift from feature drift alone.

PSI bands (<0.10, 0.10–0.25, >=0.25) and the default performance RMSE threshold are starting heuristics, not universal service-level objectives. Calibrate them against historical data and business costs.

## Deployment safety

The gate does not overwrite `artifacts/metadata.json`, copy model files into a live pod, or deploy to GKE. Register the approved candidate and invoke the existing CI/CD/deployment workflow as a distinct promotion step. This prevents Airflow from silently replacing the model being served.
