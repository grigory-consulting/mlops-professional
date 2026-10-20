"""Drift-Maße für skalare Signale und für Embeddings.

Verglichen wird immer ein Referenzfenster (Anfragen aus einer Zeit, in der das
System nachweislich gut lief) mit einem aktuellen Fenster. Alle Funktionen
brauchen keine Labels und laufen deshalb auch dann, wenn die Wahrheit erst
Wochen später eintrifft.

Skalare Signale (Textlänge, Konfidenz, Top-1-Ähnlichkeit): `psi`.
Embeddings: `centroid_distance`, `mmd_rbf`, `domain_classifier_auc`.
Alarmschwelle aus driftfreien Stichproben: `null_threshold`.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index eines skalaren Signals.

    Die Klassengrenzen sind die Quantile der Referenz: Jede Klasse enthält dort
    etwa gleich viele Werte, das Maß hängt dann nicht an der Skala des Signals.
    eps verhindert log(0), wenn eine Klasse im aktuellen Fenster leer bleibt.
    Faustregel: unter 0,1 stabil, 0,1 bis 0,25 beobachten, über 0,25 handeln.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    inner = edges[1:-1]  # Randklassen bleiben offen: neue Extremwerte fallen hinein
    p = np.bincount(np.searchsorted(inner, reference, side="right"), minlength=len(inner) + 1) / len(reference)
    q = np.bincount(np.searchsorted(inner, current, side="right"), minlength=len(inner) + 1) / len(current)
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def centroid_distance(ref_emb: np.ndarray, cur_emb: np.ndarray) -> float:
    """Cosine-Distanz der beiden Mittelpunkte (0 = gleiche Richtung).

    Billig und anschaulich, sieht aber nur Verschiebungen des Schwerpunkts.
    Der Wert ist auch ohne Drift größer als 0 (Stichprobenrauschen) und sinkt
    mit der Fenstergröße: Schwellen gelten nur für eine feste Fenstergröße.
    """
    a, b = ref_emb.mean(axis=0), cur_emb.mean(axis=0)
    return float(1.0 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _subsample(emb: np.ndarray, max_samples: int | None, rng: np.random.Generator) -> np.ndarray:
    if max_samples is None or len(emb) <= max_samples:
        return emb
    return emb[rng.choice(len(emb), size=max_samples, replace=False)]


def _sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (a * a).sum(axis=1)[:, None] + (b * b).sum(axis=1)[None, :] - 2.0 * (a @ b.T)
    return np.maximum(d, 0.0)


def median_gamma(emb: np.ndarray, max_samples: int = 500, seed: int = 0) -> float:
    """Bandbreite des RBF-Kerns per Median-Heuristik: gamma = 1 / Median(||x - y||^2)."""
    x = _subsample(np.asarray(emb, dtype=np.float64), max_samples, np.random.default_rng(seed))
    d = _sq_dists(x, x)
    return float(1.0 / np.median(d[np.triu_indices_from(d, k=1)]))


def mmd_rbf(
    ref_emb: np.ndarray,
    cur_emb: np.ndarray,
    gamma: float | None = None,
    max_samples: int = 2000,
    seed: int = 0,
) -> float:
    """Maximum Mean Discrepancy (quadriert, erwartungstreuer Schätzer) mit RBF-Kern.

    Vergleicht ganze Verteilungen: mittlere Ähnlichkeit innerhalb der Referenz
    plus innerhalb des aktuellen Fensters minus zweimal die Ähnlichkeit über
    Kreuz. Ohne Drift liegt der Wert um 0 (darf leicht negativ sein).
    gamma=None nimmt die Median-Heuristik auf der Referenz; so bleibt die
    Bandbreite über alle Wochenfenster gleich. Die Kernmatrix wächst
    quadratisch, deshalb höchstens max_samples Punkte je Seite.
    """
    rng = np.random.default_rng(seed)
    x = _subsample(np.asarray(ref_emb, dtype=np.float64), max_samples, rng)
    y = _subsample(np.asarray(cur_emb, dtype=np.float64), max_samples, rng)
    if gamma is None:
        gamma = median_gamma(x, seed=seed)
    kxx, kyy, kxy = (np.exp(-gamma * _sq_dists(a, b)) for a, b in ((x, x), (y, y), (x, y)))
    m, n = len(x), len(y)
    # Diagonale (k(x, x) = 1) weglassen: sonst ist der Schätzer nach oben verzerrt
    within_x = (kxx.sum() - m) / (m * (m - 1))
    within_y = (kyy.sum() - n) / (n * (n - 1))
    return float(within_x + within_y - 2.0 * kxy.mean())


def domain_classifier_auc(
    ref_emb: np.ndarray,
    cur_emb: np.ndarray,
    folds: int = 5,
    max_samples: int | None = 2000,
    seed: int = 0,
) -> float:
    """AUC eines Klassifikators, der Referenz und aktuelles Fenster trennen soll.

    0,5 heißt: nicht unterscheidbar, kein Drift. 1,0 heißt: perfekt trennbar.
    Die AUC stammt aus einer Kreuzvalidierung, sonst würde das Modell bei 384
    Dimensionen und wenigen hundert Punkten die Fenster auswendig trennen.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    rng = np.random.default_rng(seed)
    x_ref = _subsample(np.asarray(ref_emb), max_samples, rng)
    x_cur = _subsample(np.asarray(cur_emb), max_samples, rng)
    x = np.vstack([x_ref, x_cur])
    y = np.r_[np.zeros(len(x_ref), dtype=int), np.ones(len(x_cur), dtype=int)]
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    proba = cross_val_predict(LogisticRegression(C=1.0, max_iter=500), x, y, cv=cv, method="predict_proba")
    return float(roc_auc_score(y, proba[:, 1]))


def embedding_drift(ref_emb: np.ndarray, cur_emb: np.ndarray, gamma: float | None = None, seed: int = 0) -> dict[str, float]:
    """Alle drei Embedding-Maße für ein Fensterpaar."""
    return {
        "centroid_distance": centroid_distance(ref_emb, cur_emb),
        "mmd": mmd_rbf(ref_emb, cur_emb, gamma=gamma, seed=seed),
        "domain_auc": domain_classifier_auc(ref_emb, cur_emb, seed=seed),
    }


def null_threshold(
    measure: Callable[[np.ndarray, np.ndarray], float],
    reference: np.ndarray,
    pool: np.ndarray,
    window_size: int,
    n_boot: int = 200,
    sigmas: float = 3.0,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    """Alarmschwelle aus driftfreien Stichproben: Mittelwert + sigmas * Streuung.

    Zieht n_boot Fenster der Größe window_size aus pool (Daten ohne Drift,
    Embeddings oder ein skalares Signal) und misst jedes mit
    measure(reference, window). Die Schwelle beschreibt, wie groß das Maß
    allein durch Zufall wird. Sie gilt nur für diese Fenstergröße.
    Negative sigmas liefern eine untere Schwelle (für Signale, die bei
    Problemen fallen, z. B. die mittlere Konfidenz).
    Liefert (Schwelle, gemessene Nullwerte).
    """
    rng = np.random.default_rng(seed)
    values = np.array([
        measure(reference, pool[rng.choice(len(pool), size=window_size, replace=False)])
        for _ in range(n_boot)
    ])
    return float(values.mean() + sigmas * values.std(ddof=1)), values
