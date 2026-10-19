"""Bausteine für Lab 1 (Demo und Übungen): Embeddings cachen, Varianten einbetten, je Query bewerten.

Die Funktionen ergänzen `common.metrics` und `common.models`, ohne deren
Verhalten zu ändern:

- `cached_array`, `corpus_embeddings`, `query_embeddings`: Embeddings einmal
  rechnen und unter labs/.cache/ ablegen. Der Dateiname enthält Modell und
  Fingerabdruck des Eval-Sets, damit ein Modell- oder Datenwechsel nie still
  alte Vektoren liefert.
- `encode_variant`, `unnormalized_encoder`: dieselben Gewichte mit anderen
  Einstellungen (max_seq_length, Pooling, ohne Normierung), für die
  Skew-Messungen aus Kapitel 3.
- `ndcg_per_query`: nDCG@k je Query statt nur des Mittelwerts (für Bootstrap).
- `evaluate_encoder`: ein Modell auf dem Eval-Set bewerten, Ergebnis passend
  zu `mlflow.log_params` und `mlflow.log_metrics`.
- `paired_bootstrap`: Konfidenzintervall der Differenz zweier Modelle.
- `package_versions`, `plot_quality_throughput`, `hits_table`.
- `quick_test_variants`, `classification_protocols`: Vergleichstabellen, die
  die Übung nach den Aufgaben A und E ausgibt.
"""
from __future__ import annotations

import os
import tempfile
from importlib import metadata
from typing import Callable

import numpy as np
import pandas as pd

from .evalset import EvalSet
from .metrics import retrieval_metrics, top_k
from .models import EMBEDDING_MODELS, encode, load_encoder, size_mb, timed_encode
from .paths import CACHE_DIR

PACKAGES = ("sentence-transformers", "transformers", "torch", "mlflow", "scikit-learn", "numpy", "pandas")


def cached_array(name: str, compute: Callable[[], np.ndarray]) -> np.ndarray:
    """Liest labs/.cache/<name>.npy. Fehlt die Datei, wird einmal gerechnet und abgelegt.

    Geschrieben wird erst in eine temporäre Datei und dann umbenannt: Ein
    abgebrochener Lauf oder ein paralleler Leser sieht nie eine halbe Datei.
    """
    path = CACHE_DIR / f"{name}.npy"
    if path.exists():
        return np.load(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    array = np.asarray(compute())
    fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".npy.tmp")
    with os.fdopen(fd, "wb") as handle:
        np.save(handle, array)
    os.replace(tmp, path)
    return array


def corpus_embeddings(short_name: str, evalset: EvalSet, model=None, batch_size: int = 64) -> np.ndarray:
    """Normierte Embeddings des Korpus, gecacht als lab1_<modell>_corpus_<fingerabdruck>.npy."""
    def compute() -> np.ndarray:
        return encode(model if model is not None else load_encoder(short_name), evalset.corpus, batch_size)

    return cached_array(f"lab1_{short_name}_corpus_{evalset.fingerprint}", compute)


def query_embeddings(short_name: str, evalset: EvalSet, model=None, batch_size: int = 64) -> np.ndarray:
    """Normierte Embeddings der Queries, gecacht als lab1_<modell>_queries_<fingerabdruck>.npy."""
    def compute() -> np.ndarray:
        return encode(model if model is not None else load_encoder(short_name), evalset.queries, batch_size)

    return cached_array(f"lab1_{short_name}_queries_{evalset.fingerprint}", compute)


def encode_variant(
    model,
    texts: list[str],
    max_seq_length: int | None = None,
    pooling: str | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Bettet mit geänderten Einstellungen ein und stellt das Modell danach wieder her.

    max_seq_length: Tokens je Text einschließlich der Sondertokens, der Rest wird abgeschnitten.
    pooling: "mean" oder "cls" (Modul 1 eines SentenceTransformer ist das Pooling).
    """
    old_length, old_pooling = model.max_seq_length, model[1].pooling_mode
    try:
        if max_seq_length is not None:
            model.max_seq_length = max_seq_length
        if pooling is not None:
            model[1].pooling_mode = pooling
        return encode(model, texts, batch_size)
    finally:
        model.max_seq_length, model[1].pooling_mode = old_length, old_pooling


def unnormalized_encoder(model):
    """Dieselben Gewichte ohne Normalize-Schicht: Transformer und Pooling, sonst nichts."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(modules=[model[0], model[1]], device=str(model.device))


def encode_unnormalized(model, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Rohe Vektoren, so wie Transformer und Pooling sie liefern (Länge ungleich 1)."""
    raw = unnormalized_encoder(model)
    emb = raw.encode(texts, batch_size=batch_size, normalize_embeddings=False,
                     convert_to_numpy=True, show_progress_bar=False)
    return emb.astype(np.float32)


def ndcg_per_query(
    query_emb: np.ndarray,
    query_labels: np.ndarray,
    corpus_emb: np.ndarray,
    corpus_labels: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """nDCG@k je Query (binäre Relevanz). Der Mittelwert ist ndcg_at_k aus retrieval_metrics."""
    ranked = top_k(query_emb, corpus_emb, k)
    rel = (corpus_labels[ranked] == query_labels[:, None]).astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    n_relevant = np.bincount(corpus_labels, minlength=int(corpus_labels.max()) + 1)[query_labels]
    ideal = np.array([discounts[: min(k, int(n))].sum() for n in n_relevant])
    return np.divide((rel * discounts).sum(axis=1), ideal, out=np.zeros(len(rel)), where=ideal > 0)


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 1, level: float = 0.95
) -> tuple[float, float, float]:
    """Gepaarter Bootstrap über die Queries: (Differenz b - a, untere, obere Grenze).

    Gepaart heißt: Jede Ziehung nimmt für beide Modelle dieselben Queries. So
    fällt die Schwierigkeit der einzelnen Query heraus, nur der Unterschied
    der Modelle bleibt.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = (b - a)[idx].mean(axis=1)
    tail = (1.0 - level) / 2 * 100
    low, high = np.percentile(diffs, [tail, 100 - tail])
    return float((b - a).mean()), float(low), float(high)


def evaluate_encoder(
    short_name: str,
    evalset: EvalSet,
    data_hash: str,
    k: int = 10,
    batch_size: int = 64,
    device: str = "cpu",
) -> tuple[dict, dict]:
    """Bewertet ein Embedding-Modell auf dem Eval-Set: (Parameter, Metriken) für MLflow.

    Der Korpus kommt aus dem Cache (Indexierung vorab), die Queries werden frisch
    eingebettet und gemessen (Anfragepfad). Ein Warmlauf vorher hält den
    Kaltstart aus der Durchsatzmessung heraus.
    """
    model = load_encoder(short_name, device=device)
    corpus = corpus_embeddings(short_name, evalset, model=model, batch_size=batch_size)
    encode(model, evalset.queries[:batch_size], batch_size)  # Warmlauf
    queries, seconds = timed_encode(model, evalset.queries, batch_size)
    params = {
        "model": short_name,
        "hub_id": EMBEDDING_MODELS[short_name],
        "evalset_fingerprint": evalset.fingerprint,
        "dataset_hash": data_hash,
        "n_queries": len(evalset.queries),
        "n_corpus": len(evalset.corpus),
        "k": k,
        "batch_size": batch_size,
        "device": device,
        "max_seq_length": model.max_seq_length,
        "pooling": model[1].pooling_mode,
    }
    metrics = retrieval_metrics(queries, evalset.query_labels, corpus, evalset.corpus_labels, k=k)
    metrics["texts_per_second"] = len(evalset.queries) / seconds
    metrics["size_mb"] = size_mb(model)
    return params, metrics


def hits_table(
    evalset: EvalSet, intents: list[str], query_emb: np.ndarray, corpus_emb: np.ndarray, k: int = 10
) -> pd.DataFrame:
    """Eine Zeile je Query: Text, Anliegen, nDCG@k, Hit@1 und das Anliegen des ersten Treffers."""
    ranked = top_k(query_emb, corpus_emb, k)
    first = evalset.corpus_labels[ranked[:, 0]]
    return pd.DataFrame({
        "query": evalset.queries,
        "intent": [intents[label] for label in evalset.query_labels],
        "ndcg_at_10": ndcg_per_query(query_emb, evalset.query_labels, corpus_emb, evalset.corpus_labels, k),
        "hit_at_1": (first == evalset.query_labels).astype(int),
        "top1_intent": [intents[label] for label in first],
    })


def package_versions(packages: tuple[str, ...] = PACKAGES) -> dict[str, str]:
    """Installierte Versionen der Pakete, die das Ergebnis beeinflussen."""
    return {package: metadata.version(package) for package in packages}


def plot_quality_throughput(results: pd.DataFrame, min_ndcg: float | None = None,
                            min_texts_per_second: float | None = None):
    """Qualität gegen Durchsatz je Modell (Fläche: Größe der Gewichte), auf Wunsch mit Gate-Schwellen.

    results braucht die Spalten model, ndcg_at_10, texts_per_second, size_mb.
    """
    import matplotlib.pyplot as plt

    from .plotstyle import ACCENT, INK, WARN

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.scatter(results["texts_per_second"], results["ndcg_at_10"], s=results["size_mb"] * 4,
               color=ACCENT, alpha=0.85, edgecolor=INK, zorder=3)
    for row in results.itertuples():
        ax.annotate(f"{row.model}\n{row.size_mb:.0f} MB", (row.texts_per_second, row.ndcg_at_10),
                    xytext=(0, 22), textcoords="offset points", ha="center", color=INK, fontsize=10)
    if min_ndcg is not None:
        ax.axhline(min_ndcg, color=WARN, linestyle="--", linewidth=1.2)
        ax.text(0.99, min_ndcg - 0.008, f"Gate: nDCG@10 ab {min_ndcg:.2f}".replace(".", ","), color=WARN,
                ha="right", va="top", fontsize=10, transform=ax.get_yaxis_transform())
    if min_texts_per_second is not None:
        ax.axvline(min_texts_per_second, color=WARN, linestyle="--", linewidth=1.2)
        ax.text(min_texts_per_second, 0.02, f" Mindestdurchsatz: {min_texts_per_second:.0f} Texte/s",
                color=WARN, ha="left", va="bottom", fontsize=10, transform=ax.get_xaxis_transform())
    ax.set_xlim(0, results["texts_per_second"].max() * 1.2)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Durchsatz (Texte je Sekunde, CPU)")
    ax.set_ylabel("Qualität (nDCG@10)")
    ax.set_title("Embedding-Modelle auf Banking77: Qualität gegen Durchsatz")
    return fig


def quick_test_variants(quick_test: Callable[..., float], quick_set: EvalSet, quick_metrics: dict,
                        evalset: EvalSet, query_emb: np.ndarray, corpus_emb: np.ndarray,
                        train: pd.DataFrame, test: pd.DataFrame, n_intents: int, seed: int = 1) -> pd.DataFrame:
    """Vergleichstabelle: Varianten des Schnelltests neben dem Eval-Set des Kurses.

    quick_test ist die Funktion `schnelltest` der Übung, query_emb und corpus_emb sind die
    Vektoren des Eval-Sets. Druckt vorab, ob die Zeilen mit 200 Anfragen dieselben Anfragen nutzen.
    """
    def per_label(labels) -> str:
        counts = np.bincount(labels, minlength=n_intents)
        missing = int((counts == 0).sum())
        return f"{counts.min()} bis {counts.max()}" + (f" ({missing} Anliegen fehlen)" if missing else "")

    official = retrieval_metrics(query_emb, evalset.query_labels, corpus_emb, evalset.corpus_labels)
    labels = train["label"].to_numpy()
    table = pd.DataFrame([
        ("Schnelltest der Kollegin (Seed 1)", quick_test(seed=seed), 3000, per_label(labels[:3000]), 200,
         "ohne Fingerabdruck"),
        ("Schnelltest, ganzer Korpus (Seed 1)", quick_test(n_corpus=None, seed=seed), len(train), per_label(labels),
         200, "ohne Fingerabdruck"),
        ("Schnelltest repariert", quick_metrics["hit_at_1"], len(quick_set.corpus), per_label(quick_set.corpus_labels),
         len(quick_set.queries), quick_set.fingerprint),
        ("Eval-Set des Kurses (Demo 5)", official["hit_at_1"], len(evalset.corpus), per_label(evalset.corpus_labels),
         len(evalset.queries), evalset.fingerprint),
    ], columns=["Variante", "Hit@1", "Dokumente", "Beispiele je Anliegen", "Anfragen", "Eval-Set"]).set_index("Variante")
    table.insert(1, "Standardfehler", np.sqrt(table["Hit@1"] * (1 - table["Hit@1"]) / table["Anfragen"]))

    sampled = test.sample(200, random_state=seed).sort_index()["text"].tolist()  # die Anfragen von quick_test(seed)
    print("Die drei Zeilen mit 200 Anfragen nutzen dieselben 200 Testanfragen:", sampled == quick_set.queries)
    print(f"Davon stehen im Eval-Set des Kurses: {len(set(sampled) & set(evalset.queries))} von 200")
    return table


def classification_protocols(few_shot_accuracy: Callable[..., float], X_train: np.ndarray, X_test: np.ndarray,
                             y_train: np.ndarray, y_test: np.ndarray, acc_card: float, acc_mteb: float):
    """Vergleichstabelle: dieselben Vektoren unter mehreren Klassifikationsprotokollen.

    few_shot_accuracy ist die Funktion der Übung. Liefert (Tabelle, Accuracy normiert mit C=1,
    mittlere Länge der rohen Vektoren).
    """
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    lengths = np.linalg.norm(X_train, axis=1, keepdims=True)
    X_train_unit, X_test_unit = X_train / lengths, X_test / np.linalg.norm(X_test, axis=1, keepdims=True)
    c_scaled = float(lengths.mean() ** 2)
    acc_unit = few_shot_accuracy(X_train_unit, X_test_unit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        acc_all = LogisticRegression(C=10, max_iter=100).fit(X_train_unit, y_train).score(X_test_unit, y_test)
    table = pd.Series({
        "Model Card (MTEB)": acc_card,
        "roh, 8 je Anliegen, C=1 (MTEB-Protokoll)": acc_mteb,
        "normiert, 8 je Anliegen, C=1": acc_unit,
        f"normiert, 8 je Anliegen, C={c_scaled:.0f}": few_shot_accuracy(X_train_unit, X_test_unit, C=c_scaled),
        "normiert, alle Trainingsdaten, C=10": acc_all,
    }, name="Accuracy")
    return table, acc_unit, float(lengths.mean())
