# Ordner labs/ finden und zum Arbeitsordner machen: Das Paket common und die Daten liegen dort.
# Nötig, wenn der Kernel woanders startet (PyCharm, VS Code mit Arbeitsordner im Pack-Root, Quelle in labs/_src/).
import os, sys
from pathlib import Path
for _p in [Path.cwd(), Path.cwd() / "labs", *Path.cwd().parents]:
    if (_p / "common" / "__init__.py").exists():
        os.chdir(_p)
        sys.path.insert(0, str(_p))
        break
else:
    raise RuntimeError("Ordner labs/ nicht gefunden. Öffnen Sie das Notebook aus dem Ordner labs/ des Kursmaterials.")

import json
import logging
import time
import warnings

warnings.filterwarnings("ignore", message=".*IProgress not found.*")

import mlflow
import numpy as np
import pandas as pd

from common.paths import ARTIFACTS_DIR, offline

offline()  # vor dem ersten Laden eines Modells: die Labs laufen ohne Netzzugriff

from common.contract import GERMAN_QUERIES, NEW_DELIVERY, check_contract, known_word_share
from common.data import dataset_hash, load_banking77
from common.encoder_eval import (corpus_embeddings, hits_table, ndcg_per_query, package_versions,
                                 paired_bootstrap, query_embeddings)
from common.evalset import build_evalset
from common.interactive import explore
from common.metrics import retrieval_metrics, top_k
from common.models import EMBEDDING_MODELS, encode, load_encoder, size_mb, timed_encode
from common.plotstyle import use_course_style
from common.tracking import setup_mlflow

use_course_style()
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)  # Hinweis beim Laden von rohem DistilBERT
SEED, DEVICE, BATCH_SIZE = 1, "cpu", 64
OUT_DIR = ARTIFACTS_DIR / "lab1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.disable(logging.INFO)  # das erste Anlegen der Datenbank schreibt sonst viele Migrationszeilen
tracking_uri = setup_mlflow("lab1")
logging.disable(logging.NOTSET)
logging.getLogger("mlflow").setLevel(logging.ERROR)

train, test, intents = load_banking77()
print(f"{len(train)} Trainings- und {len(test)} Testanfragen, {len(intents)} Anliegen")





minilm = load_encoder("minilm-l6", device=DEVICE)
distilbert = load_encoder("distilbert-raw", device=DEVICE)

a = "I am still waiting on my card?"   # card_arrival
b = "My card isn't here yet."          # card_arrival
c = "What exchange rate is it?"        # exchange_rate
ANFRAGE_AUS_DEM_RAUM = "My new card has still not arrived"  # hier eine Anfrage aus der Gruppe eintragen

table = {}
for name, model in [("minilm-l6", minilm), ("distilbert-raw", distilbert)]:
    emb = encode(model, [a, b, c, ANFRAGE_AUS_DEM_RAUM])
    sim = emb @ emb.T  # normierte Vektoren: Skalarprodukt = Cosine
    table[name] = {
        "a, b (gleiches Anliegen)": sim[0, 1],
        "a, c (anderes Anliegen)": sim[0, 2],
        "Abstand": sim[0, 1] - sim[0, 2],
        "Raum, a": sim[3, 0],
        "Raum, c": sim[3, 2],
    }
cosines = pd.DataFrame(table)
print(f"Größe: minilm-l6 {size_mb(minilm):.0f} MB, distilbert-raw {size_mb(distilbert):.0f} MB")
cosines.round(3)
