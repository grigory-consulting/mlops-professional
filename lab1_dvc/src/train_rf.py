"""DVC Stage 3: RandomForest trainieren, in MLflow loggen, Metriken schreiben."""
import json
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = yaml.safe_load((ROOT / "params.yaml").read_text())

    X_train = np.load(ROOT / "data" / "X_train.npy")
    X_test = np.load(ROOT / "data" / "X_test.npy")
    y_train = np.load(ROOT / "data" / "y_train.npy")
    y_test = np.load(ROOT / "data" / "y_test.npy")

    tracking_uri = p["mlflow"]["tracking_uri"]
    if tracking_uri.startswith("file:./") or tracking_uri.startswith("./"):
        rel = tracking_uri.removeprefix("file:")
        tracking_uri = f"file:{(ROOT / rel).resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(p["mlflow"]["experiment"])

    with mlflow.start_run(run_name="dvc-repro"):
        mlflow.log_params({
            "n_estimators": p["rf"]["n_estimators"],
            "max_depth": p["rf"]["max_depth"],
            "seed": p["seed"],
        })
        mlflow.set_tag("source", "dvc-pipeline")

        model = RandomForestClassifier(
            n_estimators=p["rf"]["n_estimators"],
            max_depth=p["rf"]["max_depth"],
            random_state=p["seed"],
            n_jobs=-1,
        ).fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        }
        mlflow.log_metrics(metrics)

        sig = infer_signature(X_train[:5], model.predict(X_train[:5]))
        mlflow.sklearn.log_model(
            sk_model=model, artifact_path="model",
            signature=sig, input_example=X_train[:5],
        )

    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ROOT / "models" / "rf.pkl")

    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Metriken:", metrics)


if __name__ == "__main__":
    main()
