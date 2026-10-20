from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from airflow.sdk import dag, task

PIPELINE_DIR = Path(__file__).resolve().parent.parent
LAB_PYTHON = os.environ.get("LAB_PYTHON", str(PIPELINE_DIR.parent.parent / ".venv" / "bin" / "python"))
OFFLINE = "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"

def stage(script: str) -> str:
    return f"cd {PIPELINE_DIR} && {OFFLINE} {LAB_PYTHON} {script}"

def pipeline():
    return f"cd {PIPELINE_DIR} && dvc repro"

@dag(
    dag_id="banking_intent_pipeline",
    schedule="@weekly",              # Retraining im Wochentakt; ein Drift-Alarm kann ihn zusätzlich anstoßen
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,               # alle Läufe schreiben in dieselben Pfade unter PIPELINE_DIR
    tags=["mlops-kurs", "banking77"],
    default_args={"retries": 1},
)
def banking_intent_pipeline():



    @task.bash
    def validate_data() -> str:
        return stage("validate_data.py")

    @task.bash
    def train() -> str:
        return stage("train.py")

    @task.bash
    def evaluate() -> str:
        return stage("evaluate.py")

    @task.bash
    def gate() -> str:
        return stage("gate.py")

    @task.short_circuit
    def gate_not_failed() -> bool:
        status = json.loads((PIPELINE_DIR / "reports" / "gate.json").read_text())["status"]
        print(f"Gate-Status: {status}")
        return status != "fail"

    @task.bash
    def register() -> str:
        return stage("register.py")



    validate_data() >> train() >> evaluate() >> gate() >> gate_not_failed() >> register()


banking_intent_pipeline()