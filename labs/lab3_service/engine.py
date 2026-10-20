"""Kern des Service ohne Web-Schicht: Konfiguration, Vektorindex, Klassifikator, Prüfungen.

Derselbe Code schreibt den Index (Batch-Pfad, im Notebook) und liest ihn
(Anfragepfad, im Service). So gibt es nur eine Stelle, die das Format kennt.

Das Modul importiert bewusst nichts aus `labs/common/`: Der Container enthält
nur diesen Ordner, die Modelle und den Index.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

DEFAULT_ENCODER = "minilm-l6"
# So viele Indexeinträge bettet der Service beim Start neu ein und vergleicht
# sie mit den gespeicherten Vektoren.
PROBE_SIZE = 16
PROBE_MIN_COSINE = 0.99

EncodeFn = Callable[[list[str]], np.ndarray]


class IndexMismatchError(RuntimeError):
    """Encoder und Index passen nicht zusammen (Versions-Skew)."""


# --------------------------------------------------------------------------
# Konfiguration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """Alle Stellschrauben des Service. Quelle sind Umgebungsvariablen.

    LAB_BASE_DIR     Wurzel mit `models/` und `artifacts/` (Standard: labs/)
    ENCODER_PATH     Ordner eines SentenceTransformer-Modells
    ENCODER_NAME     Kurzname des Encoders (Standard: Name des Ordners)
    INDEX_DIR        Ordner des Vektorindex (Standard: artifacts/lab3/index_<ENCODER_NAME>)
    CLASSIFIER_PATH  Ordner eines Hugging-Face-Klassifikators
    """

    base_dir: Path
    encoder_path: Path
    encoder_name: str
    encoder_source: str
    index_dir: Path
    classifier_path: Path | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        base_dir = Path(env.get("LAB_BASE_DIR") or Path(__file__).resolve().parent.parent)
        if env.get("ENCODER_PATH"):
            encoder_path, source = Path(env["ENCODER_PATH"]), "env"
        else:
            encoder_path, source = base_dir / "models" / DEFAULT_ENCODER, "default"
        encoder_name = env.get("ENCODER_NAME") or encoder_path.name
        # Ohne INDEX_DIR hängt der Index am Namen des Encoders: Wer nur ENCODER_PATH
        # umstellt, bekommt den passenden Index (oder einen klaren Fehler). Steht
        # INDEX_DIR fest im Deployment, bleibt der alte Index stehen; dann fängt erst
        # check_compatibility beim Start das falsche Paar.
        index_dir = Path(env.get("INDEX_DIR") or base_dir / "artifacts" / "lab3" / f"index_{encoder_name}")
        classifier_path = Path(env["CLASSIFIER_PATH"]) if env.get("CLASSIFIER_PATH") else None
        return cls(base_dir, encoder_path, encoder_name, source, index_dir, classifier_path)


# --------------------------------------------------------------------------
# Vektorindex
# --------------------------------------------------------------------------
def index_fingerprint(embeddings: np.ndarray, texts: list[str], intents: list[str], model_name: str) -> str:
    """Kurzer SHA-256 über Vektoren, Texte, Anliegen und Modellname."""
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(np.ascontiguousarray(embeddings, dtype=np.float32).tobytes())
    for text, intent in zip(texts, intents):
        h.update(text.encode("utf-8"))
        h.update(b"\x1f")
        h.update(intent.encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()[:16]


@dataclass
class VectorIndex:
    """Exakte Suche per Skalarprodukt auf L2-normierten Vektoren (Cosine-Similarity).

    Für 10 000 Vektoren mit 384 Dimensionen ist das eine Matrixmultiplikation
    von unter einer Millisekunde. Eine Vektordatenbank ändert an den Aussagen
    dieses Labs nichts: Auch dort gehört zu jedem Index genau ein Modell.
    """

    embeddings: np.ndarray
    texts: list[str]
    intents: list[str]
    meta: dict

    @property
    def model_name(self) -> str:
        return self.meta["model_name"]

    @property
    def fingerprint(self) -> str:
        return self.meta["fingerprint"]

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Liefert (Position, Score) der k ähnlichsten Einträge, absteigend sortiert."""
        scores = self.embeddings @ query_vec
        k = min(k, len(scores))
        part = np.argpartition(-scores, kth=k - 1)[:k]
        order = part[np.argsort(-scores[part])]
        return [(int(i), float(scores[i])) for i in order]


def write_index(
    index_dir: Path,
    embeddings: np.ndarray,
    texts: list[str],
    intents: list[str],
    model_name: str,
    data_hash: str,
    evalset_fingerprint: str = "",
) -> dict:
    """Schreibt embeddings.npy, docs.json und meta.json. Liefert die Metadaten."""
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if not (len(embeddings) == len(texts) == len(intents)):
        raise ValueError("Vektoren, Texte und Anliegen müssen gleich viele Einträge haben.")
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_name": model_name,
        "dim": int(embeddings.shape[1]),
        "size": int(embeddings.shape[0]),
        "normalized": True,
        "data_hash": data_hash,
        "evalset_fingerprint": evalset_fingerprint,
        "fingerprint": index_fingerprint(embeddings, texts, intents, model_name),
    }
    np.save(index_dir / "embeddings.npy", embeddings)
    (index_dir / "docs.json").write_text(
        json.dumps({"texts": texts, "intents": intents}, ensure_ascii=False), encoding="utf-8"
    )
    (index_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_index(index_dir: Path, verify: bool = True) -> VectorIndex:
    """Liest einen Index und prüft, ob Dateien und Metadaten zusammengehören."""
    index_dir = Path(index_dir)
    if not (index_dir / "meta.json").exists():
        raise FileNotFoundError(f"Kein Index unter {index_dir}. Bitte zuerst den Index bauen (Batch-Pfad, write_index).")
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    docs = json.loads((index_dir / "docs.json").read_text(encoding="utf-8"))
    # allow_pickle=False: Eine .npy-Datei darf hier nur Zahlen enthalten, keinen Code.
    embeddings = np.load(index_dir / "embeddings.npy", allow_pickle=False)
    index = VectorIndex(embeddings, docs["texts"], docs["intents"], meta)
    if verify:
        actual = index_fingerprint(embeddings, index.texts, index.intents, meta["model_name"])
        if actual != meta["fingerprint"] or embeddings.shape != (meta["size"], meta["dim"]):
            raise IndexMismatchError(
                f"Index unter {index_dir} ist beschädigt: Fingerabdruck {actual} statt {meta['fingerprint']}."
            )
    return index


def check_compatibility(encoder_name: str, encode: EncodeFn, index: VectorIndex) -> float:
    """Schutz vor Versions-Skew: Passt dieser Encoder zu diesem Index?

    Stufe 1 vergleicht die Namen (billig, aber nur eine Behauptung).
    Stufe 2 bettet die ersten PROBE_SIZE Indexeinträge neu ein und vergleicht
    sie mit den gespeicherten Vektoren (Schwelle PROBE_MIN_COSINE).
    Liefert die kleinste Cosine-Similarity der Stichprobe.
    """
    if index.model_name != encoder_name:
        raise IndexMismatchError(
            f"Index wurde mit '{index.model_name}' gebaut, der Service lädt den Encoder '{encoder_name}'. "
            "Vektoren verschiedener Modelle sind nicht vergleichbar. "
            "Bitte INDEX_DIR auf den Index dieses Encoders setzen oder den Index neu bauen."
        )
    probe_texts = index.texts[:PROBE_SIZE]
    probe = encode(probe_texts)
    if probe.shape[1] != index.embeddings.shape[1]:
        raise IndexMismatchError(
            f"Encoder liefert {probe.shape[1]} Dimensionen, der Index hat {index.embeddings.shape[1]}."
        )
    cosine = float(np.sum(probe * index.embeddings[: len(probe_texts)], axis=1).min())
    if cosine < PROBE_MIN_COSINE:
        raise IndexMismatchError(
            f"Der Encoder '{encoder_name}' reproduziert die Vektoren im Index nicht "
            f"(kleinste Cosine-Similarity der Stichprobe: {cosine:.3f}, verlangt: {PROBE_MIN_COSINE}). "
            "Der Name stimmt, das Modell dahinter aber nicht."
        )
    return cosine


# --------------------------------------------------------------------------
# Klassifikator: zwei Wege, eine Schnittstelle
# --------------------------------------------------------------------------
def write_logreg(path: Path, coef: np.ndarray, intercept: np.ndarray, intents: list[str], encoder_name: str) -> None:
    """Speichert die Gewichte einer logistischen Regression als .npz (nur Zahlen und Texte, kein Pickle)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        coef=np.asarray(coef, dtype=np.float32),
        intercept=np.asarray(intercept, dtype=np.float32),
        intents=np.array(intents),
        encoder_name=np.array(encoder_name),
    )


class LogRegClassifier:
    """Ersatzweg: logistische Regression auf den Embeddings des Encoders."""

    source = "fallback-logreg"

    def __init__(self, path: Path, encoder_name: str):
        data = np.load(path, allow_pickle=False)
        trained_on = str(data["encoder_name"])
        if trained_on != encoder_name:
            # Auch der Klassifikatorkopf hängt am Encoder: gleiche Regel wie beim Index.
            raise IndexMismatchError(
                f"Klassifikator {Path(path).name} wurde auf Embeddings von '{trained_on}' trainiert, "
                f"geladen ist '{encoder_name}'."
            )
        self.coef, self.intercept = data["coef"], data["intercept"]
        self.intents = [str(i) for i in data["intents"]]
        self.name = f"logreg-on-{encoder_name}"

    def predict_proba(self, texts: list[str], encode: EncodeFn) -> np.ndarray:
        logits = encode(texts) @ self.coef.T + self.intercept
        logits -= logits.max(axis=1, keepdims=True)
        prob = np.exp(logits)
        return prob / prob.sum(axis=1, keepdims=True)


class HFClassifier:
    """Fein abgestimmtes BERT-Modell aus einem Hugging-Face-Ordner (Ergebnis von Lab 2)."""

    def __init__(self, path: Path, source: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path)).to("cpu").eval()
        id2label = self.model.config.id2label
        self.intents = [str(id2label[i]) for i in range(len(id2label))]
        self.name = f"{self.model.config.model_type}:{Path(path).name}"
        self.source = source

    def predict_proba(self, texts: list[str], encode: EncodeFn) -> np.ndarray:
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with self._torch.inference_mode():
            logits = self.model(**batch).logits
        return self._torch.softmax(logits, dim=-1).numpy()


def load_classifier(settings: Settings):
    """Reihenfolge: CLASSIFIER_PATH, dann Ergebnis von Lab 2, dann der Ersatzweg."""
    lab2_dir = settings.base_dir / "artifacts" / "lab2" / "classifier"
    if settings.classifier_path is not None:
        return HFClassifier(settings.classifier_path, source="env")
    if (lab2_dir / "config.json").exists():
        return HFClassifier(lab2_dir, source="lab2")
    fallback = settings.base_dir / "artifacts" / "lab3" / f"logreg_{settings.encoder_name}.npz"
    if not fallback.exists():
        raise FileNotFoundError(
            f"Kein Klassifikator gefunden: weder CLASSIFIER_PATH noch {lab2_dir} noch {fallback}."
        )
    return LogRegClassifier(fallback, settings.encoder_name)


# --------------------------------------------------------------------------
# Engine: alles, was die Endpunkte brauchen
# --------------------------------------------------------------------------
class Engine:
    def __init__(self, settings: Settings, encoder, index: VectorIndex, classifier):
        self.settings = settings
        self.encoder = encoder
        self.index = index
        self.classifier = classifier
        self.dim = int(index.embeddings.shape[1])
        # Ein Modell, ein Lock: Tokenizer und Modell sind nicht für parallele
        # Aufrufe aus mehreren Threads gebaut.
        self._lock = threading.Lock()

    def embed(self, texts: list[str]) -> np.ndarray:
        """L2-normierte Embeddings, derselbe Aufruf wie beim Bau des Index."""
        with self._lock:
            emb = self.encoder.encode(
                texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            )
        return emb.astype(np.float32)

    def classify(self, text: str) -> dict:
        prob = self.classifier.predict_proba([text], self.embed)[0]
        top = np.argsort(-prob)[:3]
        return {
            "intent": self.classifier.intents[int(top[0])],
            "confidence": float(prob[top[0]]),
            "top3": [{"intent": self.classifier.intents[int(i)], "score": float(prob[i])} for i in top],
            "model": self.classifier.name,
            "source": self.classifier.source,
        }

    def similar(self, text: str, k: int) -> list[dict]:
        query_vec = self.embed([text])[0]
        return [
            {"rank": rank, "text": self.index.texts[i], "intent": self.index.intents[i], "score": score}
            for rank, (i, score) in enumerate(self.index.search(query_vec, k), start=1)
        ]

    def health(self) -> dict:
        meta = self.index.meta
        return {
            "encoder": {"name": self.settings.encoder_name, "source": self.settings.encoder_source, "dim": self.dim},
            "classifier": {
                "name": self.classifier.name,
                "source": self.classifier.source,
                "n_intents": len(self.classifier.intents),
            },
            "index": {
                "model_name": meta["model_name"],
                "size": meta["size"],
                "dim": meta["dim"],
                "fingerprint": meta["fingerprint"],
                "data_hash": meta["data_hash"],
            },
        }


def load_engine(settings: Settings, check: bool = True) -> Engine:
    """Lädt Encoder, Index und Klassifikator.

    `check=False` schaltet den Schutz ab. Das gibt es nur, um in den Notebooks
    einen Service ohne Prüfung beim Start nachzustellen, nie im Betrieb.
    """
    from sentence_transformers import SentenceTransformer

    if not settings.encoder_path.exists():
        raise FileNotFoundError(f"Encoder-Ordner fehlt: {settings.encoder_path}")
    encoder = SentenceTransformer(str(settings.encoder_path), device="cpu")
    index = load_index(settings.index_dir)
    engine = Engine(settings, encoder, index, classifier=None)
    if check:
        check_compatibility(settings.encoder_name, engine.embed, index)
    engine.classifier = load_classifier(settings)
    return engine
