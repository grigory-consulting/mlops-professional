"""Pfade des Lab-Ordners. Alle Labs laufen offline aus diesen Verzeichnissen."""
from __future__ import annotations

import os
from pathlib import Path

LABS_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = LABS_DIR / "data" / "banking77"
MODELS_DIR = LABS_DIR / "models"
CACHE_DIR = LABS_DIR / ".cache"
ARTIFACTS_DIR = LABS_DIR / "artifacts"


def offline() -> None:
    """Schaltet Hugging Face auf den lokalen Bestand um (kein Netzzugriff im Lab)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
