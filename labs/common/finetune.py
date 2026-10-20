"""Bausteine für Lab 2: Baseline, Fine-Tuning mit MLflow-Kurven, Bewertung, Gate, Ablage, Registry.

Demo und Übungen von Lab 2 rufen dieselben Funktionen auf. Das Notebook öffnet
den MLflow-Run selbst (`with mlflow.start_run(...)`), die Funktionen hier
schreiben in den gerade aktiven Run: Parameter, Kurven (`train_loss`,
`val_accuracy`) und Kennzahlen. Ohne aktiven Run loggen sie nichts.

Zuschnitt für CPU: `bert-mini` (4 Schichten, 256 Dimensionen, rund 11 Mio.
Parameter), `max_length` 32 Token, 1 bis 3 Epochen.

Weitere Bausteine: `linear_probe` (logistische Regression auf eingefrorenen
Vektoren), `load_prepared` (vorbereiteten Lauf lesen, ohne zu schreiben),
`rule_gate` (Gate über mehrere Kennzahlen); `save_classifier` tauscht den
Zielordner erst, wenn er vollständig geschrieben ist.
"""
from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .data import dataset_hash
from .models import encode, load_encoder, model_path, size_mb
from .paths import CACHE_DIR
from .tracking import CLASSIFIER_NAME

SEED = 1
BASE_MODEL = "bert-mini"
MAX_LENGTH = 32
MIN_ACCURACY = 0.85   # Mindestqualität für das Routing
TOLERANCE = 0.02      # so viel schlechter als die Baseline darf ein Modell automatisch durchgehen
EXAMPLES = ["I am still waiting on my card?", "What is the exchange rate for euros today?"]


# --------------------------------------------------------------------------
# Daten und Baseline
# --------------------------------------------------------------------------
def split_fit_val(train: pd.DataFrame, val_fraction: float = 0.1, seed: int = SEED,
                  fit_fraction: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratifizierter Validierungssplit. fit_fraction < 1 zieht zusätzlich eine stratifizierte Stichprobe.

    Entscheidungen (Hyperparameter, Modellwahl) fallen nur auf den
    Validierungsdaten; das Testset bleibt bis zur Schlussbewertung unberührt.
    """
    from sklearn.model_selection import train_test_split

    fit, val = train_test_split(train, test_size=val_fraction, stratify=train["label"], random_state=seed)
    if fit_fraction < 1.0:
        fit, _ = train_test_split(fit, train_size=fit_fraction, stratify=fit["label"], random_state=seed)
    return fit, val


def cached_embeddings(encoder_name: str, frame: pd.DataFrame, encoder=None,
                      refresh: bool = False) -> tuple[np.ndarray, float | None]:
    """Embeddings eines DataFrames, zwischengespeichert in labs/.cache/.

    Der Dateiname trägt Modell und Fingerabdruck der Texte. Liefert
    (Embeddings, Sekunden); Sekunden ist None, wenn der Cache gegriffen hat.
    refresh=True rechnet neu und überschreibt die Datei.
    """
    path = CACHE_DIR / f"lab2_{encoder_name}_{dataset_hash(frame)[:16]}.npy"
    if path.exists() and not refresh:
        return np.load(path), None
    encoder = encoder or load_encoder(encoder_name)
    start = time.perf_counter()
    emb = encode(encoder, frame["text"].tolist())
    seconds = time.perf_counter() - start
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb, seconds


def baseline_logreg(fit_df: pd.DataFrame, val_df: pd.DataFrame, test: pd.DataFrame,
                    encoder_name: str = "minilm-l6", C: float = 10.0, use_cache: bool = True) -> dict:
    """Logistische Regression auf den Embeddings eines Encoders, der selbst nicht trainiert wird.

    use_cache=False rechnet die Embeddings frisch (und misst so die ehrliche
    Gesamtzeit); das Ergebnis landet trotzdem im Cache für spätere Aufrufe.
    Loggt in den aktiven MLflow-Run, falls einer offen ist.
    """
    import mlflow
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    encoder = None if use_cache else load_encoder(encoder_name)
    fresh = encoder is not None
    emb_fit, encode_fit = cached_embeddings(encoder_name, fit_df, encoder, refresh=fresh)
    emb_val, _ = cached_embeddings(encoder_name, val_df, encoder, refresh=fresh)
    emb_test, encode_test = cached_embeddings(encoder_name, test, encoder, refresh=fresh)

    start = time.perf_counter()
    head = LogisticRegression(C=C, max_iter=1000).fit(emb_fit, fit_df["label"])
    fit_seconds = time.perf_counter() - start
    pred = head.predict(emb_test)
    result = {
        "val_accuracy": float(accuracy_score(val_df["label"], head.predict(emb_val))),
        "test_accuracy": float(accuracy_score(test["label"], pred)),
        "test_macro_f1": float(f1_score(test["label"], pred, average="macro")),
        "fit_seconds": fit_seconds,
    }
    if encode_fit is not None:
        result["train_seconds"] = encode_fit + fit_seconds        # Embeddings rechnen + Regression anpassen
        result["texts_per_second"] = len(test) / encode_test
    if encoder is not None:
        result["model_size_mb"] = size_mb(encoder) + (head.coef_.size + head.intercept_.size) * 8 / 1e6
    if mlflow.active_run():
        mlflow.log_params({"base_model": encoder_name, "classifier": "LogisticRegression", "C": C})
        mlflow.log_metrics(result)
    result["test_pred"] = pred
    return result


def frozen_vectors(frame: pd.DataFrame, base_model: str = BASE_MODEL, pooling: str = "pooler",
                   max_length: int = MAX_LENGTH) -> tuple[np.ndarray, float | None]:
    """Satzvektoren eines nur vortrainierten BERT, ohne Training, zwischengespeichert in labs/.cache/.

    pooling: "pooler" ist der Vektor, den der Klassifikationskopf von
    AutoModelForSequenceClassification liest ([CLS] plus dense und tanh),
    "cls" der rohe [CLS]-Vektor, "mean" der Mittelwert über die Tokens.
    Liefert (Vektoren, Sekunden); Sekunden ist None, wenn der Cache gegriffen hat.
    """
    import torch
    from transformers import AutoModel

    path = CACHE_DIR / f"lab2_{base_model}-{pooling}-{max_length}_{dataset_hash(frame)[:16]}.npy"
    if path.exists():
        return np.load(path), None
    tokenizer = load_tokenizer(base_model)
    model = AutoModel.from_pretrained(model_path(base_model)).eval()
    texts = frame["text"].tolist()
    chunks = []
    start = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(texts), 256):
            batch = tokenizer(texts[begin:begin + 256], truncation=True, max_length=max_length, padding=True,
                              return_tensors="pt")
            out = model(**batch)
            if pooling == "pooler":
                vectors = out.pooler_output
            elif pooling == "cls":
                vectors = out.last_hidden_state[:, 0]
            elif pooling == "mean":
                mask = batch["attention_mask"].unsqueeze(-1)
                vectors = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            else:
                raise ValueError(f"pooling {pooling!r}: erlaubt sind pooler, cls, mean")
            chunks.append(vectors.numpy())
    seconds = time.perf_counter() - start
    emb = np.concatenate(chunks).astype(np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, emb)
    tmp.replace(path)  # erst umbenennen, wenn vollständig: parallele Leser sehen nie eine halbe Datei
    return emb, seconds


def linear_probe(fit_df: pd.DataFrame, val_df: pd.DataFrame, base_model: str = BASE_MODEL,
                 pooling: str = "pooler", C: float = 10.0) -> dict:
    """Kontrollmessung: logistische Regression bis zur Konvergenz auf eingefrorenen Vektoren.

    Gleiche Modellklasse wie der Klassifikationskopf (eine lineare Schicht mit
    Softmax), angepasst mit scikit-learn (max_iter=3000). Liefert val_accuracy, fit_seconds, encode_seconds
    (None aus dem Cache), n_iter und dim; loggt in den aktiven MLflow-Run.
    """
    import mlflow
    from sklearn.linear_model import LogisticRegression

    emb_fit, encode_seconds = frozen_vectors(fit_df, base_model, pooling)
    emb_val, _ = frozen_vectors(val_df, base_model, pooling)
    start = time.perf_counter()
    head = LogisticRegression(C=C, max_iter=3000).fit(emb_fit, fit_df["label"])
    result = {"val_accuracy": float(head.score(emb_val, val_df["label"])),
              "fit_seconds": time.perf_counter() - start, "encode_seconds": encode_seconds,
              "n_iter": int(head.n_iter_.max()), "dim": int(emb_fit.shape[1])}
    if mlflow.active_run():
        mlflow.log_params({"base_model": base_model, "pooling": pooling, "classifier": "LogisticRegression", "C": C,
                           "encoder_frozen": True})
        mlflow.log_metrics({"val_accuracy": result["val_accuracy"], "fit_seconds": result["fit_seconds"]})
    return result


# --------------------------------------------------------------------------
# Fine-Tuning
# --------------------------------------------------------------------------
def load_tokenizer(base_model: str = BASE_MODEL):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path(base_model))


def build_model(intents: list[str], base_model: str = BASE_MODEL, head_only: bool = False, seed: int = SEED):
    """Vortrainiertes BERT mit neuem, zufällig initialisiertem Kopf (77 Ausgänge).

    head_only=True friert den Encoder ein: Nur der Kopf (`classifier`) lernt.
    id2label trägt die Namen der Anliegen, damit Service und Registry
    `card_arrival` statt `LABEL_12` ausgeben.
    """
    from transformers import AutoModelForSequenceClassification, set_seed

    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path(base_model),
        num_labels=len(intents),
        id2label=dict(enumerate(intents)),
        label2id={name: idx for idx, name in enumerate(intents)},
    )
    if head_only:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(("classifier", "pre_classifier"))
    return model


class _IntentDataset:
    """Tokenisierte Anfragen mit Labels; aufgefüllt wird erst je Batch (dynamisches Padding)."""

    def __init__(self, tokenizer, frame: pd.DataFrame, max_length: int):
        self.encodings = tokenizer(frame["text"].tolist(), truncation=True, max_length=max_length)
        self.labels = [int(label) for label in frame["label"]]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {"input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.labels[idx]}


def _compute_metrics(eval_pred) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    pred = eval_pred.predictions.argmax(axis=-1)
    return {"accuracy": accuracy_score(eval_pred.label_ids, pred),
            "macro_f1": f1_score(eval_pred.label_ids, pred, average="macro")}


def _log_to_mlflow(result: dict) -> None:
    """Parameter, Kurven und Kennzahlen in den aktiven Run (falls offen)."""
    import mlflow

    if not mlflow.active_run():
        return
    mlflow.log_params(result["params"])
    for point in result["history"]:
        if "train_loss" in point:
            mlflow.log_metric("train_loss", point["train_loss"], step=point["step"])
        if "val_accuracy" in point:
            mlflow.log_metrics({"val_accuracy": point["val_accuracy"], "val_macro_f1": point["val_macro_f1"],
                                "val_loss": point["val_loss"]}, step=point["step"])
    mlflow.log_metrics({key: float(result[key]) for key in
                        ("train_seconds", "model_size_mb", "trainable_parameters")})


def finetune(fit_df: pd.DataFrame, val_df: pd.DataFrame, intents: list[str], *, learning_rate: float,
             epochs: float = 1, batch_size: int = 64, head_only: bool = False, base_model: str = BASE_MODEL,
             max_length: int = MAX_LENGTH, seed: int = SEED, evals_per_epoch: int = 4,
             output_dir: str | Path | None = None) -> dict:
    """Fine-Tuning mit dem Hugging Face Trainer, nur CPU, fester Seed.

    Liefert ein dict mit model, tokenizer, val_accuracy, val_macro_f1,
    train_seconds, trainable_parameters, model_size_mb, history (Messpunkte je
    Trainingsschritt) und params. Ist ein MLflow-Run offen, landen Parameter,
    die Kurven train_loss und val_accuracy sowie die Kennzahlen darin.
    """
    from transformers import DataCollatorWithPadding, PrinterCallback, Trainer, TrainingArguments
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    tokenizer = load_tokenizer(base_model)
    model = build_model(intents, base_model, head_only, seed)
    fit_ds = _IntentDataset(tokenizer, fit_df, max_length)
    val_ds = _IntentDataset(tokenizer, val_df, max_length)
    steps_per_epoch = math.ceil(len(fit_ds) / batch_size)
    eval_steps = max(1, math.ceil(steps_per_epoch / evals_per_epoch))
    params = {"base_model": base_model, "variant": "head_only" if head_only else "full",
              "learning_rate": learning_rate, "epochs": epochs, "batch_size": batch_size,
              "max_length": max_length, "warmup_ratio": 0.1, "seed": seed, "n_train": len(fit_df)}
    args = TrainingArguments(
        # save_strategy="no": Der Trainer legt keine Checkpoints ab, der Ordner bleibt (fast) leer
        output_dir=str(output_dir or Path(tempfile.gettempdir()) / "mlops-lab2-trainer"),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=256,
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=eval_steps,
        save_strategy="no",
        report_to="none",          # geloggt wird ausdrücklich, siehe _log_to_mlflow
        disable_tqdm=True,
        use_cpu=True,
        seed=seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=fit_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=_compute_metrics)
    trainer.remove_callback(PrinterCallback)  # keine Log-Zeilen im Notebook; die Kurven stehen in MLflow
    start = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - start
    # Schlussmessung, falls der letzte Schritt kein Messpunkt war
    evals = [e for e in trainer.state.log_history if "eval_accuracy" in e]
    if not evals or evals[-1]["step"] != trainer.state.global_step:
        trainer.evaluate()

    history: dict[int, dict] = {}
    for entry in trainer.state.log_history:
        point = history.setdefault(int(entry["step"]), {"step": int(entry["step"]),
                                                        "epoch": round(float(entry.get("epoch", 0)), 3)})
        if "loss" in entry:
            point["train_loss"] = float(entry["loss"])
        if "eval_accuracy" in entry:
            point.update(val_accuracy=float(entry["eval_accuracy"]), val_macro_f1=float(entry["eval_macro_f1"]),
                         val_loss=float(entry["eval_loss"]))
    history_list = [history[step] for step in sorted(history)]
    last = [p for p in history_list if "val_accuracy" in p][-1]
    result = {
        "model": model,
        "tokenizer": tokenizer,
        "params": params,
        "history": history_list,
        "val_accuracy": last["val_accuracy"],
        "val_macro_f1": last["val_macro_f1"],
        "train_seconds": train_seconds,
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "model_size_mb": size_mb(model),
    }
    _log_to_mlflow(result)
    return result


def _cache_key(fit_df: pd.DataFrame, val_df: pd.DataFrame, kwargs: dict) -> dict:
    return {**{k: v for k, v in kwargs.items() if k != "output_dir"},
            "data_hash": dataset_hash(fit_df, val_df)[:16]}


def load_prepared(cache_dir: str | Path, fit_df: pd.DataFrame, val_df: pd.DataFrame, keep_model: bool = True,
                  **kwargs) -> dict | None:
    """Lädt einen mit cached_finetune abgelegten Lauf, wenn Daten und Einstellungen passen, sonst None.

    Schreibt nichts in cache_dir. Gedacht für Notebooks, die einen
    vorbereiteten Lauf eines anderen Notebooks lesen (die Übung liest den
    Abschlusslauf der Demo). Kurven und Kennzahlen landen im aktiven MLflow-Run.
    """
    from transformers import AutoModelForSequenceClassification

    cache_dir = Path(cache_dir)
    info_path = cache_dir / "finetune_info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info["key"] != json.loads(json.dumps(_cache_key(fit_df, val_df, kwargs))):
        return None
    if keep_model and not (cache_dir / "model.safetensors").exists():
        return None
    model = AutoModelForSequenceClassification.from_pretrained(cache_dir) if keep_model else None
    tokenizer = load_tokenizer(info["params"]["base_model"]) if keep_model else None
    result = {**info["result"], "model": model, "tokenizer": tokenizer,
              "params": info["params"], "history": info["history"], "from_cache": True}
    _log_to_mlflow(result)
    return result


def cached_finetune(cache_dir: str | Path, fit_df: pd.DataFrame, val_df: pd.DataFrame, intents: list[str],
                    keep_model: bool = True, **kwargs) -> dict:
    """Wie finetune, aber mit Ablage: Liegt in cache_dir ein Lauf mit denselben Einstellungen, wird er geladen.

    Gedacht für Läufe, die für eine Live-Vorführung zu lange dauern. Kurven
    und Kennzahlen stammen dann aus dem gespeicherten Lauf, der Schlüssel
    `from_cache` sagt, ob gerechnet oder geladen wurde. keep_model=False
    speichert nur Kurven und Kennzahlen (z. B. für HPO-Trials), model ist
    beim Laden dann None. Frisch rechnen: Ordner löschen, Zelle erneut ausführen.
    """
    cache_dir = Path(cache_dir)
    key = _cache_key(fit_df, val_df, kwargs)
    loaded = load_prepared(cache_dir, fit_df, val_df, keep_model=keep_model, **kwargs)
    if loaded is not None:
        return loaded
    result = finetune(fit_df, val_df, intents, **kwargs)
    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True)
    if keep_model:
        result["model"].save_pretrained(cache_dir, safe_serialization=True)
    scalars = {k: result[k] for k in ("val_accuracy", "val_macro_f1", "train_seconds",
                                      "trainable_parameters", "model_size_mb")}
    info_path = cache_dir / "finetune_info.json"
    info_path.write_text(json.dumps({"key": key, "params": result["params"], "history": result["history"],
                                     "result": scalars, "measured_at": time.strftime("%Y-%m-%d")},
                                    indent=2) + "\n", encoding="utf-8")
    result["from_cache"] = False
    return result


# --------------------------------------------------------------------------
# Bewertung, Gate, Ablage, Registry
# --------------------------------------------------------------------------
def evaluate(model, tokenizer, frame: pd.DataFrame, max_length: int = MAX_LENGTH, batch_size: int = 256) -> dict:
    """Accuracy, Macro-F1 und Durchsatz auf einem DataFrame (in der Regel dem Testset).

    Loggt test_accuracy, test_macro_f1 und texts_per_second in den aktiven Run.
    """
    import mlflow
    import torch
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    texts = frame["text"].tolist()
    chunks = []
    start = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(texts), batch_size):
            batch = tokenizer(texts[begin:begin + batch_size], truncation=True, max_length=max_length,
                              padding=True, return_tensors="pt")
            chunks.append(model(**batch).logits.argmax(dim=-1).numpy())
    seconds = time.perf_counter() - start
    pred = np.concatenate(chunks)
    result = {"test_accuracy": float(accuracy_score(frame["label"], pred)),
              "test_macro_f1": float(f1_score(frame["label"], pred, average="macro")),
              "texts_per_second": len(texts) / seconds}
    if mlflow.active_run():
        mlflow.log_metrics(result)
    result["test_pred"] = pred
    return result


def quality_gate(accuracy: float, baseline_accuracy: float, min_accuracy: float = MIN_ACCURACY,
                 tolerance: float = TOLERANCE) -> dict:
    """Metric-Gate mit drei Ausgängen, dieselbe Regel wie die Stufe gate der DVC-Pipeline.

    fail    Accuracy unter der Mindestqualität oder kein gültiger Wert (NaN)
    review  Mindestqualität erreicht, aber mehr als tolerance unter der Baseline: Ein Mensch entscheidet
    pass    Mindestqualität erreicht und nicht schlechter als Baseline minus tolerance
    """
    # Jeder Vergleich mit NaN ergibt False: Ohne diese Prüfung käme ein NaN als pass durch.
    if not (math.isfinite(accuracy) and math.isfinite(baseline_accuracy)):
        return {"status": "fail", "reason": f"Accuracy {accuracy} oder Baseline {baseline_accuracy} ist kein gültiger Wert"}
    if accuracy < min_accuracy:
        return {"status": "fail", "reason": f"Accuracy {accuracy:.3f} unter Mindestqualität {min_accuracy:.2f}"}
    if accuracy < baseline_accuracy - tolerance:
        return {"status": "review",
                "reason": f"Accuracy {accuracy:.3f} liegt {(baseline_accuracy - accuracy) * 100:.1f} Punkte unter "
                          f"der Baseline {baseline_accuracy:.3f} (Toleranz {tolerance * 100:.0f} Punkte)"}
    return {"status": "pass", "reason": f"Accuracy {accuracy:.3f} erfüllt Mindestqualität und Baseline-Regel"}


def rule_gate(candidate: dict, baseline: dict, rules: dict[str, tuple[float, float]]) -> dict:
    """Gate über mehrere Kennzahlen, jede mit (Mindestwert, Toleranz gegen die Baseline).

    fail    eine Kennzahl fehlt oder ist ungültig (None, NaN, inf), bei Kandidat
            oder Baseline, oder der Kandidat liegt unter dem Mindestwert
    review  alle Mindestwerte erreicht, aber eine Kennzahl liegt mehr als die
            Toleranz unter der Baseline: Ein Mensch entscheidet
    pass    sonst
    Liefert {"status": ..., "reasons": [...]}; reasons nennt jede verletzte Regel.
    """
    def valid(value) -> bool:
        return isinstance(value, (int, float, np.floating)) and math.isfinite(value)

    invalid = [name for name in rules if not (valid(candidate.get(name)) and valid(baseline.get(name)))]
    if invalid:
        return {"status": "fail", "reasons": [f"{name} fehlt oder ist kein gültiger Wert" for name in invalid]}
    too_low = [f"{name} {candidate[name]:.3f} unter dem Mindestwert {minimum:.2f}"
               for name, (minimum, _) in rules.items() if candidate[name] < minimum]
    if too_low:
        return {"status": "fail", "reasons": too_low}
    behind = [f"{name} {candidate[name]:.3f} liegt {(baseline[name] - candidate[name]) * 100:.1f} Punkte unter der "
              f"Baseline {baseline[name]:.3f} (Toleranz {tolerance * 100:.0f} Punkte)"
              for name, (_, tolerance) in rules.items() if candidate[name] < baseline[name] - tolerance]
    return {"status": "review" if behind else "pass", "reasons": behind}


def save_classifier(model, tokenizer, out_dir: str | Path, metadata: dict | None = None,
                    max_length: int = MAX_LENGTH) -> Path:
    """Speichert den Klassifikator als Hugging-Face-Ordner (safetensors, Tokenizer, config mit id2label).

    Diesen Ordner lädt der Service in Lab 3 (Umgebungsvariable CLASSIFIER_PATH).
    Ein pickle-Checkpoint (pytorch_model.bin) entsteht nicht: Er könnte beim
    Laden beliebigen Code ausführen.
    """
    out_dir = Path(out_dir)
    # Erst vollständig in einen Nachbarordner schreiben, dann tauschen: Wer den Ordner
    # gleichzeitig liest (Service, andere Notebooks), sieht nie einen halben Stand.
    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    old_dir = out_dir.with_name(out_dir.name + ".old")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tokenizer.model_max_length = max_length  # der Service kürzt dann wie das Training
    model.save_pretrained(tmp_dir, safe_serialization=True)
    tokenizer.save_pretrained(tmp_dir)
    (tmp_dir / "training_metadata.json").write_text(
        json.dumps({"max_length": max_length, **(metadata or {})}, indent=2) + "\n", encoding="utf-8")
    config = json.loads((tmp_dir / "config.json").read_text(encoding="utf-8"))
    assert (tmp_dir / "model.safetensors").exists() and not (tmp_dir / "pytorch_model.bin").exists()
    assert len(config["id2label"]) == model.config.num_labels
    shutil.rmtree(old_dir, ignore_errors=True)
    if out_dir.exists():
        out_dir.rename(old_dir)
    tmp_dir.rename(out_dir)
    shutil.rmtree(old_dir, ignore_errors=True)
    return out_dir


def register_classifier(model_dir: str | Path, run_id: str, name: str = CLASSIFIER_NAME) -> str:
    """Loggt den Ordner als mlflow.transformers-Modell im Run run_id und legt eine neue Registry-Version an.

    pip_requirements schreibt die Versionen fest (sonst pinnt MLflow 2.22
    zusätzlich torchvision). Liefert die Versionsnummer als Text.
    """
    import logging

    import mlflow
    import torch
    import transformers
    from transformers import pipeline

    for noisy in ("mlflow.transformers", "mlflow.utils.environment", "mlflow.models.model"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    classifier = pipeline("text-classification", model=str(model_dir), tokenizer=str(model_dir), device="cpu")
    with mlflow.start_run(run_id=run_id), warnings.catch_warnings():
        # MLflow 2.22 ruft intern eine eigene, als veraltet markierte Funktion auf
        warnings.filterwarnings("ignore", category=FutureWarning)
        info = mlflow.transformers.log_model(
            transformers_model=classifier,
            artifact_path="classifier",
            input_example=EXAMPLES,
            pip_requirements=[f"transformers=={transformers.__version__}",
                              f"torch=={torch.__version__.split('+')[0]}",
                              f"mlflow=={mlflow.__version__}"],
            registered_model_name=name,
        )
    return str(info.registered_model_version)
