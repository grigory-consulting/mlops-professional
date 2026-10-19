"""Lädt Datensatz und Modelle einmalig in den Lab-Ordner (braucht Netzzugang).

Aufruf (im Ordner labs/):
    python prefetch.py            # Daten und alle Modelle
    python prefetch.py --check    # nur prüfen, was vorhanden ist

Danach laufen alle Labs offline:
    labs/data/banking77/   train.parquet, test.parquet, intents.json
    labs/models/<name>/    Gewichte (safetensors), Tokenizer, Konfiguration

Die Revisionen sind fest gepinnt, damit alle Teilnehmenden dieselben Dateien
und damit dieselben Kennzahlen bekommen.
"""
from __future__ import annotations

import json
import sys

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from common.models import CLASSIFIER_BASE, EMBEDDING_MODELS, SHARED_FOLDER
from common.paths import DATA_DIR, MODELS_DIR

# Banking77 (Casanueva et al. 2020, PolyAI, CC BY 4.0) in der Parquet-Fassung des MTEB-Projekts
DATASET_REPO = "mteb/banking77"
DATASET_REVISION = "18072d2685ea682290f7b8924d94c62acc19c0b2"

# Formate, die die Labs nicht brauchen (spart rund zwei Drittel des Downloads)
SKIP = ["onnx/*", "openvino/*", "coreml/*", "*.onnx", "*.h5", "*.ot", "*.msgpack",
        "tf_model*", "flax_model*", "rust_model*", "*.tflite"]


# Ordnername -> Hub-ID; Modelle mit gemeinsamem Ordner werden nur einmal geladen
FOLDERS = {
    SHARED_FOLDER.get(name, name): repo
    for name, repo in {**EMBEDDING_MODELS, **CLASSIFIER_BASE}.items()
}


def fetch_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frames = {}
    for split in ("train", "test"):
        path = hf_hub_download(
            DATASET_REPO, f"data/{split}-00000-of-00001.parquet",
            repo_type="dataset", revision=DATASET_REVISION,
        )
        df = pd.read_parquet(path).rename(columns={"label_text": "intent"})
        frames[split] = df[["text", "label", "intent"]].reset_index(drop=True)
        frames[split].to_parquet(DATA_DIR / f"{split}.parquet", index=False)
    labels = pd.concat(frames.values()).drop_duplicates("label").sort_values("label")
    intents = labels["intent"].tolist()
    (DATA_DIR / "intents.json").write_text(json.dumps(intents, indent=2), encoding="utf-8")
    print(f"Daten: {len(frames['train'])} Training, {len(frames['test'])} Test, {len(intents)} Anliegen")


def fetch_model(short_name: str, repo_id: str) -> None:
    target = MODELS_DIR / short_name
    files = HfApi().list_repo_files(repo_id)
    skip = list(SKIP)
    if "model.safetensors" in files:
        skip.append("pytorch_model.bin")  # safetensors genügt und führt keinen Code aus
    snapshot_download(repo_id, local_dir=target, ignore_patterns=skip)
    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1e6
    print(f"Modell {short_name:15s} {repo_id:45s} {size:6.0f} MB")


def check() -> bool:
    ok = True
    for name in ("train.parquet", "test.parquet", "intents.json"):
        present = (DATA_DIR / name).exists()
        ok &= present
        print(f"{'ok    ' if present else 'FEHLT '} data/banking77/{name}")
    for short_name in FOLDERS:
        present = (MODELS_DIR / short_name / "config.json").exists()
        ok &= present
        print(f"{'ok    ' if present else 'FEHLT '} models/{short_name}")
    return ok


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(0 if check() else 1)
    fetch_data()
    for short_name, repo_id in FOLDERS.items():
        fetch_model(short_name, repo_id)
    check()
