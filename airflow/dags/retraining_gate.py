"""Fail-closed scheduled retraining gate.

This DAG intentionally does not overwrite the serving model or deploy to GKE.
Candidate training/evaluation must produce the two JSON reports first; approved
promotion is a separate deployment action after artifact registration.
"""
from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

PROJECT_DIR = os.environ.get("PROJECT_DIR", "/opt/predictive-maintenance")
CANDIDATE_REPORT = os.environ.get("CANDIDATE_REPORT", "artifacts/candidates/latest/metadata.json")
CHAMPION_REPORT = os.environ.get("CHAMPION_REPORT", "artifacts/metadata.json")

with DAG(
    dag_id="predictive_maintenance_retraining_gate",
    description="Validate candidate metrics before any model promotion",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={"owner": "ml-platform", "retries": 0, "retry_delay": timedelta(minutes=5)},
    tags=["mlops", "retraining", "approval-gate"],
) as dag:
    check_candidate = BashOperator(
        task_id="evaluate_promotion_gate",
        bash_command=(
            f"cd {PROJECT_DIR} && python -m monitoring.retraining_gate "
            f"--candidate {CANDIDATE_REPORT} --champion {CHAMPION_REPORT} "
            "--output monitoring/reports/retraining_gate.json"
        ),
    )
    approved = EmptyOperator(task_id="candidate_approved", trigger_rule="all_success")
    check_candidate >> approved
