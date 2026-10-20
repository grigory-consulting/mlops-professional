"""Stufe 5: Modell in der MLflow Model Registry ablegen, abhängig vom Gate.

Eingabe:  model/, reports/gate.json, reports/train_metrics.json, params.yaml (Abschnitt registry)
Ausgabe:  reports/registration.json
    fail    keine Registrierung
    review  neue Version mit Alias candidate (Freigabe bleibt eine Entscheidung von Menschen)
    pass    neue Version mit den Aliasen candidate und champion
"""
import pipeline_lib as lib


def main() -> None:
    cfg = lib.load_params()["registry"]
    gate = lib.read_json("reports/gate.json")
    train_metrics = lib.read_json("reports/train_metrics.json")
    result = {"registered": False, "gate_status": gate["status"], "model_name": cfg["model_name"], "aliases": []}

    if gate["status"] != "fail":
        # Schwere Importe erst hier: Bei einem fail bleibt die Stufe in Sekundenbruchteilen fertig.
        import logging
        import os

        os.environ.setdefault("MLFLOW_DEFAULT_PREDICTION_DEVICE", "cpu")
        import mlflow
        import torch
        import transformers
        from mlflow import MlflowClient
        from mlflow.models import infer_signature
        from transformers import pipeline

        from common.tracking import setup_mlflow

        logging.getLogger("mlflow").setLevel(logging.ERROR)
        transformers.utils.logging.set_verbosity_error()
        setup_mlflow(cfg["experiment"])
        classifier = pipeline("text-classification", model="model", tokenizer="model", device="cpu")
        example = ["I am still waiting on my card?"]
        with mlflow.start_run(run_id=train_metrics["run_id"]):
            info = mlflow.transformers.log_model(
                transformers_model=classifier,
                artifact_path="classifier",
                signature=infer_signature(example, classifier(example)),
                input_example=example,
                pip_requirements=[f"transformers=={transformers.__version__}",
                                  f"torch=={torch.__version__.split('+')[0]}", f"mlflow=={mlflow.__version__}"],
                registered_model_name=cfg["model_name"],
            )
        client = MlflowClient()
        version = info.registered_model_version
        client.set_model_version_tag(cfg["model_name"], version, "gate_status", gate["status"])
        client.set_model_version_tag(cfg["model_name"], version, "data_hash", train_metrics["data_hash"])
        aliases = ["candidate"] + (["champion"] if gate["status"] == "pass" else [])
        for alias in aliases:
            client.set_registered_model_alias(cfg["model_name"], alias, version)
        result.update({"registered": True, "version": int(version), "aliases": aliases,
                       "run_id": train_metrics["run_id"]})

    lib.write_json("reports/registration.json", result)
    print(f"register: {result}")


if __name__ == "__main__":
    main()
