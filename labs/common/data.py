"""Datenmodul: Banking77 laden und einen Fingerabdruck der Daten bilden.

Banking77 (Casanueva et al. 2020, CC BY 4.0): 13 083 Kundenanfragen einer Bank,
77 Anliegen. Die Parquet-Dateien liegen nach `python prefetch.py` unter
labs/data/banking77/ und werden ohne Netzzugriff gelesen.
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from .paths import DATA_DIR


def load_banking77() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Liefert (train, test, intents). Spalten: text, label (int), intent (str)."""
    if not (DATA_DIR / "train.parquet").exists():
        raise FileNotFoundError(
            f"{DATA_DIR} fehlt. Bitte einmal mit Netzzugang `python prefetch.py` ausführen."
        )
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    intents = json.loads((DATA_DIR / "intents.json").read_text(encoding="utf-8"))
    return train, test, intents


def dataset_hash(*frames: pd.DataFrame) -> str:
    """SHA-256 über Texte und Labels, unabhängig von Pandas-Version und Index.

    Der Hash gehört in jeden Run: Zwei Läufe sind nur vergleichbar, wenn sie
    auf denselben Daten gerechnet wurden.
    """
    h = hashlib.sha256()
    for df in frames:
        for text, label in zip(df["text"], df["label"]):
            h.update(text.encode("utf-8"))
            h.update(b"\x1f")
            h.update(str(int(label)).encode("ascii"))
            h.update(b"\x1e")
    return h.hexdigest()
