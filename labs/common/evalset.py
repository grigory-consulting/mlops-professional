"""Evaluations-Set für das Embedding-Modell.

Aufgabe: Zu einer neuen Kundenanfrage (Query) sollen frühere Anfragen mit
demselben Anliegen gefunden werden. Korpus sind die Trainingsanfragen, Queries
eine feste Stichprobe der Testanfragen. Relevant ist ein Dokument genau dann,
wenn es dasselbe Anliegen trägt wie die Query.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvalSet:
    queries: list[str]
    query_labels: np.ndarray
    corpus: list[str]
    corpus_labels: np.ndarray
    seed: int

    @property
    def fingerprint(self) -> str:
        """Kurzer Hash über Queries, Korpus und Labels: die Version des Eval-Sets."""
        h = hashlib.sha256()
        for texts, labels in ((self.queries, self.query_labels), (self.corpus, self.corpus_labels)):
            for text, label in zip(texts, labels):
                h.update(text.encode("utf-8"))
                h.update(b"\x1f")
                h.update(str(int(label)).encode("ascii"))
                h.update(b"\x1e")
        return h.hexdigest()[:16]


def build_evalset(train: pd.DataFrame, test: pd.DataFrame, n_queries: int = 1000, seed: int = 1) -> EvalSet:
    """Zieht n_queries Testanfragen (fester Seed) und nimmt alle Trainingsanfragen als Korpus."""
    sample = test.sample(n=min(n_queries, len(test)), random_state=seed).sort_index()
    return EvalSet(
        queries=sample["text"].tolist(),
        query_labels=sample["label"].to_numpy(),
        corpus=train["text"].tolist(),
        corpus_labels=train["label"].to_numpy(),
        seed=seed,
    )
