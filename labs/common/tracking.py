"""MLflow-Konfiguration der Labs: eine SQLite-Datei für Runs und Registry."""
from __future__ import annotations

import os

import mlflow

from .paths import LABS_DIR

# Namen in der Model Registry, von allen Labs gemeinsam genutzt
ENCODER_NAME = "banking-encoder"
CLASSIFIER_NAME = "banking-intent-classifier"


def setup_mlflow(experiment: str) -> str:
    """Setzt Tracking-URI und Experiment. MLFLOW_TRACKING_URI hat Vorrang.

    Ohne Umgebungsvariable liegen Runs und Registry in labs/mlflow.db, die
    Artefakte in labs/mlartifacts/. Oberfläche:
        mlflow ui --backend-store-uri sqlite:///mlflow.db
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{LABS_DIR / 'mlflow.db'}")
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        artifacts = os.environ.get("MLFLOW_ARTIFACT_ROOT", str(LABS_DIR / "mlartifacts"))
        mlflow.create_experiment(experiment, artifact_location=f"file://{artifacts}/{experiment}")
    mlflow.set_experiment(experiment)
    return uri
