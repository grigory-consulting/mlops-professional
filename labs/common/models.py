"""Modelle des Kurses: lokal aus labs/models/ laden, sonst vom Hugging Face Hub."""
from __future__ import annotations

import time

import numpy as np

from .paths import MODELS_DIR

# Kurzname -> Hub-ID. Die Kurznamen sind zugleich die Ordnernamen unter labs/models/.
EMBEDDING_MODELS = {
    "minilm-l6": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
    # Gegenbeispiel: vortrainiertes BERT ohne Satz-Training, nur Mean-Pooling
    "distilbert-raw": "distilbert/distilbert-base-uncased",
}
# Kurznamen, die denselben Ordner unter labs/models/ nutzen
SHARED_FOLDER = {"distilbert-raw": "distilbert"}
CLASSIFIER_BASE = {
    "distilbert": "distilbert/distilbert-base-uncased",
    # kleines BERT (4 Schichten, 256 Dimensionen) für schnelle Läufe auf CPU
    "bert-mini": "google/bert_uncased_L-4_H-256_A-4",
}


def model_path(short_name: str) -> str:
    """Lokaler Ordner, falls vorhanden, sonst die Hub-ID."""
    local = MODELS_DIR / SHARED_FOLDER.get(short_name, short_name)
    if local.exists():
        return str(local)
    hub_ids = {**EMBEDDING_MODELS, **CLASSIFIER_BASE}
    return hub_ids.get(short_name, short_name)


def load_encoder(short_name: str, device: str = "cpu"):
    """Lädt ein Embedding-Modell als SentenceTransformer."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_path(short_name), device=device)


def encode(model, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """L2-normierte Embeddings als float32-Matrix (n, d)."""
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return emb.astype(np.float32)


def timed_encode(model, texts: list[str], batch_size: int = 64) -> tuple[np.ndarray, float]:
    """Wie encode, liefert zusätzlich die Laufzeit in Sekunden."""
    start = time.perf_counter()
    emb = encode(model, texts, batch_size)
    return emb, time.perf_counter() - start


def size_mb(model) -> float:
    """Größe der Gewichte in MB (float32: 4 Byte je Parameter)."""
    return sum(p.numel() for p in model.parameters()) * 4 / 1e6
