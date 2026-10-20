"""Stufe 2: BERT fine-tunen und als Hugging-Face-Ordner ablegen.

Eingabe:  reports/data_report.json, params.yaml (seed, data.val_fraction, train, registry.experiment)
Ausgabe:  model/ (safetensors, Tokenizer, config mit id2label), reports/train_metrics.json
Der Lauf erscheint zusätzlich als MLflow-Run im Experiment der Pipeline.
"""
import mlflow

import pipeline_lib as lib
from common.data import load_banking77
from common.tracking import setup_mlflow


def main() -> None:
    params = lib.load_params()
    cfg = params["train"]
    report = lib.read_json("reports/data_report.json")
    train, _, intents = load_banking77()
    fit_df, val_df = lib.split_train_val(train, params["data"]["val_fraction"], params["seed"], cfg["train_fraction"])

    setup_mlflow(params["registry"]["experiment"])
    with mlflow.start_run(run_name=f"pipeline-{cfg['base_model']}") as run:
        mlflow.log_params({**cfg, "seed": params["seed"], "data_hash": report["data_hash"], "n_train": len(fit_df)})
        model, tokenizer, info = lib.finetune(
            cfg["base_model"], fit_df, val_df, intents,
            learning_rate=cfg["learning_rate"], epochs=cfg["epochs"], batch_size=cfg["batch_size"],
            max_length=cfg["max_length"], seed=params["seed"],
        )
        for entry in info["history"]:
            step = int(round(entry["epoch"]))
            mlflow.log_metrics({key: value for key, value in entry.items() if key != "epoch"}, step=step)
        mlflow.log_metric("train_seconds", info["train_seconds"])

    tokenizer.model_max_length = cfg["max_length"]  # dieselbe Kürzung im Training und im Service
    model.save_pretrained("model", safe_serialization=True)
    tokenizer.save_pretrained("model")
    lib.write_json("reports/train_metrics.json", {
        "run_id": run.info.run_id,
        "data_hash": report["data_hash"],
        "n_train": info["n_train"],
        "train_seconds": round(info["train_seconds"], 1),
        "val_accuracy": info["history"][-1]["val_accuracy"],
        "val_macro_f1": info["history"][-1]["val_macro_f1"],
        "history": info["history"],
    })
    print(f"train: val_accuracy={info['history'][-1]['val_accuracy']:.4f} in {info['train_seconds']:.0f} s")


if __name__ == "__main__":
    main()
