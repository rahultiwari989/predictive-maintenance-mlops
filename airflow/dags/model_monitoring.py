"""Periodic drift and delayed-label performance checks."""
from datetime import datetime, timedelta
import os
import sys

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = os.environ.get("PROJECT_DIR", "/opt/predictive-maintenance")
sys.path.insert(0, PROJECT_DIR)

default_args = {"owner": "ml-platform", "retries": 1, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="predictive_maintenance_model_monitoring",
    description="Check feature drift and model error when delayed labels are available",
    start_date=datetime(2026, 1, 1),
    schedule="0 */6 * * *",
    catchup=False,
    default_args=default_args,
    tags=["mlops", "monitoring", "predictive-maintenance"],
) as dag:
    data_drift = BashOperator(
        task_id="check_data_drift",
        bash_command=f"cd {PROJECT_DIR} && python -m monitoring.data_drift",
        env={**os.environ},
    )
    concept_drift = BashOperator(
        task_id="check_labeled_performance",
        bash_command=f"cd {PROJECT_DIR} && python -m monitoring.concept_drift",
        env={**os.environ},
    )
    data_drift >> concept_drift
