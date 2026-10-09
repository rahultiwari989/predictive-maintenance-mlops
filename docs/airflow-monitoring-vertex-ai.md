# Airflow, model monitoring, and Vertex AI

## What this branch adds

- Structured JSON prediction logs from FastAPI, including prediction time, model version, RUL, status, and per-feature latest/mean/std values. GKE's logging pipeline can collect stdout logs.
- PSI and two-sample Kolmogorov-Smirnov feature-distribution checks.
- Delayed-label MAE/RMSE checks when verified actual RUL labels become available.
- A fail-closed candidate promotion gate requiring candidate/champion RMSE plus explicit schema and smoke-test evidence.
- Airflow DAGs: a six-hour monitoring DAG and a manually triggered promotion-gate DAG.
- Unit tests for the monitoring functions and promotion gate.

## Important integration work before production use

The code deliberately does not pretend that Airflow can see live prediction logs automatically. Configure a durable export from Cloud Logging to BigQuery or GCS, then materialize:
- monitoring/reference.csv: representative training/reference feature rows;
- monitoring/current.csv: recent production feature rows in the same feature space;
- monitoring/labeled_predictions.csv: joined predictions and verified actual RUL values once labels arrive.

Set REFERENCE_DATA, CURRENT_DATA, LABELED_PREDICTIONS, DRIFT_REPORT, and PERFORMANCE_REPORT in the Airflow worker environment. The monitoring DAG currently checks local/mounted files. For GKE, prefer Cloud Composer or a separately managed Airflow deployment rather than running a scheduler inside the inference API container.

Concept drift/performance degradation requires outcomes/labels. Feature drift is only a warning signal and must not automatically trigger deployment. PSI thresholds and the default RMSE threshold are initial heuristics; calibrate using historical data and operational costs.

The retraining gate currently consumes candidate and champion JSON metadata. The candidate training job must create a separate candidate artifact directory and report validation_rmse, schema_valid: true, and smoke_test_passed: true. Promotion/deployment remains a separate explicit workflow; this branch does not replace the live model or deploy a candidate.

## Where Vertex AI fits

Recommended incremental adoption:

1. Vertex AI Experiments — track training runs, hyperparameters, dataset identity, and validation metrics.
2. Vertex AI Model Registry — version and label champion/candidate models after the gate passes. Keep the scaler and metadata associated with each model version.
3. Vertex AI Pipelines — move repeatable training, evaluation, and registration tasks to managed pipeline components if operationally useful. Airflow/Cloud Composer can trigger a pipeline rather than duplicating its internal steps.
4. Cloud Storage — persist versioned reference datasets, candidate artifacts, drift reports, and evaluation reports.
5. Cloud Logging + BigQuery — collect structured GKE inference logs and prepare feature/prediction samples for Airflow monitoring.
6. Vertex AI Model Monitoring — evaluate only if you move serving to a supported Vertex AI Endpoint and confirm the feature schema / monitoring capabilities meet requirements. Current serving is FastAPI on GKE, so custom monitoring over GKE logs is the immediate fit.

## Suggested rollout order

1. Validate tests and DAG imports in a dedicated Airflow environment.
2. Export structured logs from Cloud Logging to BigQuery/GCS and generate current/reference samples.
3. Tune alert thresholds against historical data.
4. Make training write to versioned candidate artifacts; run the promotion gate.
5. Register approved artifacts in Vertex AI Model Registry or retain Artifact Registry/GCS-based model storage.
6. Add a human approval or protected deployment step before GKE rollout.
