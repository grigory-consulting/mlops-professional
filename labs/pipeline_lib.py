"""Gemeinsame Bausteine der Pipeline-Stufen von Lab 2.

Die Stufen-Skripte (validate_data, train, evaluate, gate, register) bleiben
dadurch kurz: Jede Stufe liest ihre Eingabedateien, ruft Funktionen von hier
und schreibt ihre Ausgabedateien. Der Trainingscode entspricht dem Notebook.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml


def find_labs_dir() -> Path:
    """Sucht den Ordner labs/ (erkennbar am Paket common/) oberhalb dieser Datei.

    Die Suche statt eines festen Pfads erlaubt es, den Pipeline-Ordner in einen
    Arbeitsordner zu kopieren (z. B. labs/artifacts/lab2/pipeline_run/).
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "common" / "paths.py").exists():
            return parent
    raise RuntimeError("Ordner labs/ mit dem Paket common/ nicht gefunden.")


LABS_DIR = find_labs_dir()
if str(LABS_DIR) not in sys.path:
    sys.path.insert(0, str(LABS_DIR))

from common.paths import offline  # noqa: E402

offline()


def load_params(path: str = "params.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def split_train_val(train, val_fraction: float, seed: int, train_fraction: float = 1.0):
    """Stratifizierter Validierungssplit; train_fraction < 1 zieht eine stratifizierte Stichprobe."""
    from sklearn.model_selection import train_test_split

    fit, val = train_test_split(train, test_size=val_fraction, stratify=train["label"], random_state=seed)
    if train_fraction < 1.0:
        fit, _ = train_test_split(fit, train_size=train_fraction, stratify=fit["label"], random_state=seed)
    return fit, val


class IntentDataset:
    """Tokenisierte Anfragen mit Labels; gepolstert wird erst je Batch (DataCollatorWithPadding)."""

    def __init__(self, tokenizer, texts: list[str], labels, max_length: int):
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length)
        self.labels = [int(label) for label in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def compute_metrics(eval_pred) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    pred = eval_pred.predictions.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(eval_pred.label_ids, pred),
        "macro_f1": f1_score(eval_pred.label_ids, pred, average="macro"),
    }


def finetune(
    base_model: str,
    fit_df,
    val_df,
    intents: list[str],
    *,
    learning_rate: float,
    epochs: float,
    batch_size: int,
    max_length: int,
    seed: int,
    train_head_only: bool = False,
    output_dir: str = "trainer_out",
):
    """Fine-Tuning mit dem Hugging Face Trainer. Liefert (model, tokenizer, info)."""
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from transformers.utils import logging as hf_logging

    from common.models import model_path

    hf_logging.set_verbosity_error()
    set_seed(seed)  # der Klassifikationskopf wird zufällig initialisiert
    tokenizer = AutoTokenizer.from_pretrained(model_path(base_model))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path(base_model),
        num_labels=len(intents),
        id2label=dict(enumerate(intents)),
        label2id={name: idx for idx, name in enumerate(intents)},
    )
    if train_head_only:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(("classifier", "pre_classifier"))

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=256,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        use_cpu=True,
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=IntentDataset(tokenizer, fit_df["text"].tolist(), fit_df["label"], max_length),
        eval_dataset=IntentDataset(tokenizer, val_df["text"].tolist(), val_df["label"], max_length),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    start = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - start
    history = [
        {"epoch": round(float(entry["epoch"]), 2), "val_accuracy": float(entry["eval_accuracy"]),
         "val_macro_f1": float(entry["eval_macro_f1"]), "val_loss": float(entry["eval_loss"])}
        for entry in trainer.state.log_history if "eval_accuracy" in entry
    ]
    info = {
        "train_seconds": train_seconds,
        "history": history,
        "n_train": len(fit_df),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    return model, tokenizer, info


def predict_labels(model, tokenizer, texts: list[str], max_length: int, batch_size: int = 256):
    """Vorhersagen für eine Liste von Texten. Liefert (labels, seconds)."""
    import torch

    model.eval()
    predictions = []
    start = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(texts), batch_size):
            batch = tokenizer(texts[begin:begin + batch_size], truncation=True, max_length=max_length,
                              padding=True, return_tensors="pt")
            predictions.append(model(**batch).logits.argmax(dim=-1).numpy())
    return np.concatenate(predictions), time.perf_counter() - start


def weights_mb(model) -> float:
    """Größe der Gewichte in MB (float32: 4 Byte je Parameter)."""
    return sum(p.numel() for p in model.parameters()) * 4 / 1e6
