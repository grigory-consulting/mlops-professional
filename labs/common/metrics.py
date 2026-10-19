"""Retrieval-Metriken auf normierten Embeddings (Skalarprodukt = Cosine-Similarity).

Relevanz ist binär: Ein Dokument ist relevant, wenn es dasselbe Anliegen trägt
wie die Query. Alle Metriken sind Mittelwerte über die Queries.
"""
from __future__ import annotations

import numpy as np


def top_k(query_emb: np.ndarray, corpus_emb: np.ndarray, k: int = 10) -> np.ndarray:
    """Indizes der k ähnlichsten Dokumente je Query, absteigend sortiert."""
    scores = query_emb @ corpus_emb.T
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def retrieval_metrics(
    query_emb: np.ndarray,
    query_labels: np.ndarray,
    corpus_emb: np.ndarray,
    corpus_labels: np.ndarray,
    k: int = 10,
) -> dict[str, float]:
    """nDCG@k, MRR@k, Precision@k und Hit@1 für binäre Relevanz."""
    ranked = top_k(query_emb, corpus_emb, k)
    rel = (corpus_labels[ranked] == query_labels[:, None]).astype(np.float64)  # (n_queries, k)

    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (rel * discounts).sum(axis=1)
    n_relevant = np.bincount(corpus_labels, minlength=int(corpus_labels.max()) + 1)[query_labels]
    ideal = np.array([discounts[: min(k, int(n))].sum() for n in n_relevant])
    ndcg = np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0)

    first_hit = np.where(rel.any(axis=1), rel.argmax(axis=1) + 1, 0)
    mrr = np.divide(1.0, first_hit, out=np.zeros(len(first_hit)), where=first_hit > 0)

    return {
        f"ndcg_at_{k}": float(ndcg.mean()),
        f"mrr_at_{k}": float(mrr.mean()),
        f"precision_at_{k}": float(rel.mean()),
        "hit_at_1": float(rel[:, 0].mean()),
    }
