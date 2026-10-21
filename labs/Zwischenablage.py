from __future__ import annotations

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
import time
import warnings
import webbrowser
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.paths import offline, ARTIFACTS_DIR, LABS_DIR, MODELS_DIR

offline()

from common import drift, modelcard, monitoring as mon
from common.data import load_banking77
from common.interactive import building, explore
from common.models import encode, load_encoder
from common.plotstyle import use_course_style, ACCENT, INK, LIGHT, MUTED, WARN

use_course_style()
warnings.filterwarnings("ignore")  # Evidently 0.4 meldet mit aktuellem pandas viele veraltete Aufrufe
SEED, WINDOW = mon.SEED, mon.WINDOW  # 500 Anfragen je Wochenfenster
OUT = ARTIFACTS_DIR / "lab4"
OUT.mkdir(parents=True, exist_ok=True)
T_START = time.perf_counter()

train, test, intents = load_banking77()
encoder = load_encoder("minilm-l6")
# Suchindex des Service: alle Anfragen aus dem Trainingsset (Embeddings beim ersten Lauf gerechnet, danach aus labs/.cache/)
index_emb = mon.cached_encode(encoder, "minilm-l6", train["text"].tolist(), "index")
index_labels = train["label"].to_numpy()
# Klassifikator aus Lab 2; fehlt er, trainiert der Ersatzweg eine logistische Regression auf den Embeddings
predict_proba, classifier_info = mon.load_classifier(index_emb, index_labels, len(intents))
print("Klassifikator:", classifier_info["source"], "|", classifier_info["kind"])
