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
from collections import Counter

import matplotlib.pyplot as plt
import mlflow
import optuna
from mlflow import MlflowClient
from sklearn.metrics import accuracy_score, f1_score

from common.paths import ARTIFACTS_DIR, LABS_DIR, offline

offline()  # vor dem ersten Laden eines Modells: die Labs laufen ohne Netzzugriff

from common import dvcrun
from common.data import dataset_hash, load_banking77
from common.finetune import (MIN_ACCURACY, SEED, TOLERANCE, baseline_logreg, cached_finetune, evaluate, finetune,
                             linear_probe, quality_gate, register_classifier, rule_gate, save_classifier,
                             split_fit_val)
from common.interactive import explore
from common.plotstyle import ACCENT, INK, LIGHT, MUTED, WARN, use_course_style
from common.tracking import CLASSIFIER_NAME, setup_mlflow

use_course_style()
optuna.logging.set_verbosity(optuna.logging.WARNING)
OUT = ARTIFACTS_DIR / "lab2"
PREPARED = OUT / "prepared"  # vorbereitete Läufe, die für die Live-Vorführung zu lange dauern
OUT.mkdir(parents=True, exist_ok=True)

logging.disable(logging.INFO)  # das erste Anlegen der Datenbank schreibt sonst viele Migrationszeilen
setup_mlflow("lab2")
logging.disable(logging.NOTSET)
client = MlflowClient()
DEMO_START = time.perf_counter()

train, test, intents = load_banking77()
fit_df, val_df = split_fit_val(train)  # 10 % stratifiziert als Validierung; das Testset bleibt bis zum Schluss liegen
DATA_HASH = dataset_hash(train, test)[:16]
print(f"Training {len(fit_df)}, Validierung {len(val_df)}, Test {len(test)}, Anliegen {len(intents)}, "
      f"Daten-Hash {DATA_HASH}")
