"""Bausteine der Drift-Überwachung für Lab 4 (Demo und Übungen).

Der Service der Bank loggt jede Anfrage. Aus diesem Log entstehen Fenster
(Arrays von Positionen im Log), und je Fenster werden Signale ohne Labels und
Embedding-Maße gegen ein festes Referenzfenster verglichen. Die Maße selbst
stehen in `common.drift`; hier liegen Datenaufbau, Fensterbau und die
Auswertung, die Demo und Übungen gemeinsam nutzen.

Aufbau des Logs (`RequestLog`): erst die unveränderten Anfragen des Pools,
dann dieselben Anfragen vertippt, dann verlängert. Eine vertippte Anfrage
liegt also an Position `i + log.offset["typo"]`.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd

from . import drift
from .models import CLASSIFIER_BASE, EMBEDDING_MODELS, encode
from .paths import ARTIFACTS_DIR, CACHE_DIR

WINDOW = 500  # Anfragen je Wochenfenster
# Acht Wochen: zwei ruhige Wochen (None = Anteil wie im Pool), danach wächst der Anteil schrittweise
CARD_SHARES_WEEKLY = (None, None, 0.40, 0.45, 0.52, 0.60, 0.70, 0.80)
TYPO_SHARES_WEEKLY = (0.0, 0.0, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60)

SEED = 1
SIGNALS = {"words": "Textlänge (Wörter)", "confidence": "Konfidenz", "top1_sim": "Top-1-Ähnlichkeit"}
MEASURE_TITLES = {"centroid_distance": "Zentroid-Distanz", "mmd": "MMD", "domain_auc": "Domain-Classifier (AUC)"}
# Klassifikator aus Lab 2, sonst Ersatzweg
CLASSIFIER_DIRS = (ARTIFACTS_DIR / "lab2" / "classifier",)


# ---------------------------------------------------------------- Embeddings und Klassifikator

def text_fingerprint(texts: list[str]) -> str:
    """Kurzer Hash über die Texte: Teil des Cache-Namens, damit nie fremde Vektoren gelesen werden."""
    h = hashlib.sha256()
    for text in texts:
        h.update(text.encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()[:16]


def cached_encode(encoder, short_name: str, texts: list[str], tag: str) -> np.ndarray:
    """Embeddings aus labs/.cache/lab4_<modell>_<tag>_<fingerabdruck>.npy, sonst einmal rechnen und ablegen."""
    path = CACHE_DIR / f"lab4_{short_name}_{tag}_{text_fingerprint(texts)}.npy"
    if path.exists():
        return np.load(path)
    emb = encode(encoder, texts)
    CACHE_DIR.mkdir(exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, emb)
    tmp.replace(path)  # erst umbenennen, wenn vollständig: parallele Leser sehen nie eine halbe Datei
    return emb


def _bert_classifier(folder: Path, n_classes: int):
    """Fein abgestimmtes Modell aus Lab 2 laden. Liefert (predict_proba, info) oder None."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(folder)
        model = AutoModelForSequenceClassification.from_pretrained(folder, use_safetensors=True).eval()
    except Exception as error:  # unvollständiger Ordner: lieber Ersatzweg als Abbruch
        print(f"Klassifikator in {folder.parent.name} nicht ladbar: {str(error)[:100]}")
        return None
    if model.config.num_labels != n_classes:
        print(f"Klassifikator in {folder.parent.name} hat {model.config.num_labels} statt {n_classes} Klassen.")
        return None
    meta_path = folder / "training_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    # Kürzen wie im Training: Der Service muss dieselbe Eingabelänge sehen wie das Modell beim Lernen
    max_length = int(meta.get("max_length", 64))
    base = meta.get("base_model") or ("distilbert" if model.config.model_type == "distilbert" else "bert-mini")
    torch.manual_seed(SEED)

    def predict_proba(texts, emb=None):
        parts = []
        with torch.no_grad():
            for i in range(0, len(texts), 128):
                batch = tokenizer(list(texts[i:i + 128]), padding=True, truncation=True,
                                  max_length=max_length, return_tensors="pt")
                parts.append(torch.softmax(model(**batch).logits, dim=-1).numpy())
        return np.vstack(parts)

    return predict_proba, {
        "source": f"Lab 2 ({folder.relative_to(ARTIFACTS_DIR.parent)})",
        "kind": f"fein abgestimmtes {model.config.model_type}-Modell mit Klassifikationskopf",
        "base_model": CLASSIFIER_BASE.get(base, base),
        "base_model_license": "Apache-2.0",
        "library_name": "transformers",
        "weights_file": str(folder / "model.safetensors"),
        "max_length": max_length,
    }


def load_classifier(index_emb: np.ndarray, index_labels: np.ndarray, n_classes: int = 77):
    """Klassifikator des Service: aus Lab 2, sonst Ersatzweg.

    Ersatzweg: logistische Regression (C=10) auf den Embeddings des Encoders,
    in etwa einer Sekunde trainiert. Liefert (predict_proba, info);
    predict_proba(texts, emb) -> Matrix (n, n_classes).
    """
    for folder in CLASSIFIER_DIRS:
        if (folder / "config.json").exists() and (folder / "model.safetensors").exists():
            loaded = _bert_classifier(folder, n_classes)
            if loaded is not None:
                return loaded
    from sklearn.linear_model import LogisticRegression

    logreg = LogisticRegression(C=10, max_iter=1000).fit(index_emb, index_labels)
    return (lambda texts, emb: logreg.predict_proba(emb)), {
        "source": "Ersatzweg (in Lab 4 trainiert)",
        "kind": "logistische Regression (C=10) auf den Embeddings des Encoders",
        "base_model": EMBEDDING_MODELS["minilm-l6"],
        "base_model_license": "Apache-2.0",
        "library_name": "sklearn",
        "weights_file": None,
        "max_length": None,
    }


# ---------------------------------------------------------------- Referenz, Varianten, Signale

def split_reference(test: pd.DataFrame, per_intent: int = 13, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Referenz (per_intent Anfragen je Anliegen) und Pool (der Rest der Testanfragen).

    Die Referenz stammt nicht aus den Trainingsanfragen: Die stehen selbst im
    Suchindex (Top-1-Ähnlichkeit 1,0) und der Klassifikator hat sie gelernt.
    """
    test = test.reset_index(drop=True)
    mask = np.zeros(len(test), dtype=bool)
    mask[test.groupby("label").sample(n=per_intent, random_state=seed).index] = True
    return test[mask].reset_index(drop=True), test[~mask].reset_index(drop=True)


def add_typos(text: str, rng: np.random.Generator, p: float = 0.3) -> str:
    """Vertippt etwa jedes dritte Wort ab vier Buchstaben (Dreher oder fehlender Buchstabe)."""
    words = []
    for word in text.split():
        if len(word) >= 4 and rng.random() < p:
            i = int(rng.integers(1, len(word) - 2))
            word = word[:i] + word[i + 1] + word[i] + word[i + 2:] if rng.random() < 0.5 else word[:i] + word[i + 1:]
        words.append(word)
    return " ".join(words)


def make_long(texts: list[str], labels: np.ndarray, rng: np.random.Generator) -> list[str]:
    """Zwei Anfragen desselben Anliegens hintereinander: länger, das Anliegen bleibt gleich."""
    long_texts = []
    for pos, (text, label) in enumerate(zip(texts, labels)):
        partners = np.flatnonzero(labels == label)
        long_texts.append(text + " " + texts[int(rng.choice(partners[partners != pos]))])
    return long_texts


def compute_signals(texts: list[str], emb: np.ndarray, labels: np.ndarray, predict_proba,
                    index_emb: np.ndarray, index_labels: np.ndarray) -> pd.DataFrame:
    """Signale je Anfrage. words, confidence, top1_sim brauchen keine Labels; correct, hit1 schon."""
    proba = predict_proba(texts, emb)
    nearest, top1 = np.empty(len(emb), dtype=int), np.empty(len(emb))
    for i in range(0, len(emb), 2048):  # stückweise: die volle Ähnlichkeitsmatrix wäre mehrere hundert MB groß
        sims = emb[i:i + 2048] @ index_emb.T
        nearest[i:i + 2048], top1[i:i + 2048] = sims.argmax(axis=1), sims.max(axis=1)
    predicted = proba.argmax(axis=1)
    return pd.DataFrame({
        "words": np.array([len(text.split()) for text in texts]),
        "confidence": proba.max(axis=1),
        "top1_sim": top1,
        # ab hier mit Labels: im Betrieb erst nutzbar, wenn die Wahrheit eingetroffen ist
        "predicted": predicted,
        "correct": predicted == labels,
        "hit1": index_labels[nearest] == labels,
    })


@dataclass
class RequestLog:
    """Das Log des Service: Texte, Labels, Embeddings und Signale je Anfrage.

    Ein Fenster ist ein Array von Positionen; log.emb[idx] und
    log.signals.iloc[idx] liefern Embeddings und Signale des Fensters.
    """
    texts: list[str]
    labels: np.ndarray
    intents: np.ndarray
    emb: np.ndarray
    signals: pd.DataFrame
    n_pool: int
    offset: dict[str, int] = field(default_factory=dict)

    @property
    def pool(self) -> np.ndarray:
        """Positionen der unveränderten Anfragen (Quelle driftfreier Fenster)."""
        return np.arange(self.n_pool)


def build_log(pool: pd.DataFrame, encoder, predict_proba, index_emb: np.ndarray, index_labels: np.ndarray,
              short_name: str = "minilm-l6", seed: int = SEED) -> RequestLog:
    """Pool unverändert, vertippt und verlängert hintereinander, mit Embeddings (gecacht) und Signalen."""
    texts, labels = pool["text"].tolist(), pool["label"].to_numpy()
    rng = np.random.default_rng(seed)
    typo_texts = [add_typos(text, rng) for text in texts]
    long_texts = make_long(texts, labels, rng)
    all_texts = texts + typo_texts + long_texts
    all_labels = np.tile(labels, 3)
    emb = cached_encode(encoder, short_name, all_texts, "log")
    signals = compute_signals(all_texts, emb, all_labels, predict_proba, index_emb, index_labels)
    n = len(texts)
    return RequestLog(all_texts, all_labels, np.tile(pool["intent"].to_numpy(), 3), emb, signals, n,
                      {"original": 0, "typo": n, "long": 2 * n})


# ---------------------------------------------------------------- Fenster

def mix_window(group: np.ndarray, rest: np.ndarray, share: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Fenster mit vorgegebenem Anteil aus group, der Rest aus rest (je ohne Zurücklegen).

    Beispiel Themenverschiebung: group = Positionen der Karten-Anliegen.
    """
    n_group = int(round(share * size))
    if n_group > len(group) or size - n_group > len(rest):
        raise ValueError(f"Zu wenige Anfragen: {n_group} aus {len(group)} und {size - n_group} aus {len(rest)} gebraucht.")
    return np.r_[rng.choice(group, n_group, replace=False), rng.choice(rest, size - n_group, replace=False)]


def variant_window(log: RequestLog, variant: str, share: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Zufälliges Fenster, in dem ein Anteil der Anfragen durch eine Variante ersetzt ist (typo, long)."""
    idx = rng.choice(log.n_pool, size, replace=False)
    idx[: int(round(share * size))] += log.offset[variant]
    return idx


def topic_groups(log: RequestLog, intents: list[str]) -> tuple[list[str], np.ndarray, np.ndarray, float]:
    """Karten-Anliegen (alle mit "card" im Namen): (Namen, Positionen Karte, Positionen übrige, Anteil im Pool)."""
    card_intents = [name for name in intents if "card" in name]
    is_card = np.isin(log.intents[: log.n_pool], card_intents)
    return card_intents, np.flatnonzero(is_card), np.flatnonzero(~is_card), float(is_card.mean())


def scenario_windows(log: RequestLog, card_pos: np.ndarray, other_pos: np.ndarray,
                     size: int = WINDOW, seed: int = SEED) -> dict[str, np.ndarray]:
    """Die vier Szenarien der Demo, je ein Fenster mit size Anfragen (feste Seeds)."""
    rng = np.random.default_rng(seed)
    return {
        "kein Drift": rng.choice(log.n_pool, size, replace=False),
        "Themenverschiebung": mix_window(card_pos, other_pos, 0.80, size, rng),
        "Tippfehler": rng.choice(log.n_pool, size, replace=False) + log.offset["typo"],
        "lange Anfragen": rng.choice(log.n_pool, size, replace=False) + log.offset["long"],
    }


def weekly_windows(log: RequestLog, card_pos: np.ndarray, other_pos: np.ndarray, base_share: float,
                   size: int = WINDOW, seed: int = SEED + 1) -> tuple[dict[str, list[np.ndarray]], dict[str, list[float]]]:
    """Zwei Reihen mit je acht Wochenfenstern: Themenverschiebung und Tippfehler. Liefert (Fenster, Anteile)."""
    rng = np.random.default_rng(seed)
    card_shares = [base_share if share is None else share for share in CARD_SHARES_WEEKLY]
    windows = {
        "Themenverschiebung": [mix_window(card_pos, other_pos, share, size, rng) for share in card_shares],
        "Tippfehler": [variant_window(log, "typo", share, size, rng) for share in TYPO_SHARES_WEEKLY],
    }
    return windows, {"Themenverschiebung": card_shares, "Tippfehler": list(TYPO_SHARES_WEEKLY)}


# ---------------------------------------------------------------- Maße, Schwellen, Alarme

def scalar_tests(reference: np.ndarray, current: np.ndarray) -> dict[str, float]:
    """PSI, KS-Test und Wasserstein-Distanz eines skalaren Signals."""
    from scipy.stats import ks_2samp, wasserstein_distance

    reference, current = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    ks = ks_2samp(reference, current)
    return {"psi": drift.psi(reference, current), "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
            "wasserstein": float(wasserstein_distance(reference, current))}


def psi_evidently(reference: np.ndarray, current: np.ndarray) -> float:
    """PSI wie in Evidently 0.4.40: gleich breite Klassen über beide Fenster zusammen (Sturges-Regel).

    Zum Vergleich mit drift.psi, das zehn Klassen aus den Quantilen der Referenz bildet.
    Leere Klassen bekommen wie bei Evidently den Anteil 0,0001.
    """
    reference, current = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    edges = np.histogram_bin_edges(np.r_[reference, current], bins="sturges")
    p = np.histogram(reference, edges)[0] / len(reference)
    q = np.histogram(current, edges)[0] / len(current)
    p, q = np.where(p == 0, 1e-4, p), np.where(q == 0, 1e-4, q)
    return float(np.sum((q - p) * np.log(q / p)))


def embedding_measures(gamma: float, seed: int = SEED) -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    """Die drei Embedding-Maße mit fester Kernbreite gamma (einmal auf der Referenz bestimmt)."""
    return {
        "centroid_distance": drift.centroid_distance,
        "mmd": lambda a, b: drift.mmd_rbf(a, b, gamma=gamma, seed=seed),
        "domain_auc": lambda a, b: drift.domain_classifier_auc(a, b, seed=seed),
    }


def measure_window(measures: dict, ref_emb: np.ndarray, cur_emb: np.ndarray) -> dict[str, float]:
    """Alle Maße für ein Fenster."""
    return {key: float(func(ref_emb, cur_emb)) for key, func in measures.items()}


def null_thresholds(measures: dict, ref_emb: np.ndarray, pool_emb: np.ndarray, window_size: int,
                    n_boot: int = 200, seed: int = SEED) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Alarmschwelle je Maß aus n_boot driftfreien Fenstern (Mittelwert + 3 Standardabweichungen).

    Gilt nur für diese Fenstergröße. Liefert (Schwellen, Nullwerte).
    """
    thresholds, values = {}, {}
    for key, func in measures.items():
        thresholds[key], values[key] = drift.null_threshold(func, ref_emb, pool_emb, window_size, n_boot=n_boot, seed=seed)
    return thresholds, values


def first_alarm(values, threshold: float, below: bool = False) -> int | None:
    """Erste Woche (ab 1), in der der Wert die Schwelle über- oder (below=True) unterschreitet."""
    values = np.asarray(values, dtype=float)
    hits = np.flatnonzero(values < threshold if below else values > threshold)
    return int(hits[0]) + 1 if len(hits) else None


def alarm_table(values: dict[str, float], thresholds: dict[str, float],
                null_values: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """Messwert, Schwelle, Ausschlag und Alarm je Maß (Ausschlag über 1 heißt Alarm).

    Ausschlag = (Wert - Mittel ohne Drift) / (Schwelle - Mittel ohne Drift), gemessen an
    den Nullwerten aus null_thresholds. Ein Verhältnis zu 0 täuscht: Die Nullwerte der MMD
    liegen um einen negativen Mittelwert, ihre Schwelle bei 500 Anfragen kaum über 0.
    Ohne null_values gilt als Basis 0 (AUC: 0,5).
    """
    rows = []
    for key, value in values.items():
        if null_values is not None:
            base = float(np.mean(null_values[key]))
        else:
            base = 0.5 if key == "domain_auc" else 0.0
        rows.append({"Maß": MEASURE_TITLES.get(key, key), "Wert": value, "Schwelle": thresholds[key],
                     "Ausschlag (Schwelle = 1)": (value - base) / (thresholds[key] - base),
                     "Alarm": value > thresholds[key]})
    return pd.DataFrame(rows).set_index("Maß")


def noise_factors(null_small: dict[str, np.ndarray], null_large: dict[str, np.ndarray]) -> dict[str, float]:
    """Wie stark das Rauschen je Maß wächst: Streuung der Nullwerte im kleinen geteilt durch die im großen Fenster.

    Bei Schwellen aus Mittelwert plus drei Standardabweichungen ist das zugleich der Faktor
    des Abstands der Schwelle vom Mittel ohne Drift.
    """
    return {key: float(np.std(null_small[key], ddof=1) / np.std(null_large[key], ddof=1)) for key in null_small}


def detection_rates(measures: dict, ref_emb: np.ndarray, emb: np.ndarray, group: np.ndarray, rest: np.ndarray,
                    shares: list[float], thresholds: dict[int, dict[str, float]], n_windows: int = 80,
                    seed: int = 2026) -> pd.DataFrame:
    """Anteil alarmierender Fenster je Fenstergröße, Anteil der Gruppe und Maß (viele Ziehungen statt einer).

    thresholds: Fenstergröße -> {Maß -> Schwelle}. Zeilen (Fenstergröße, Anteil), Spalten die Maße.
    """
    rows = {}
    for size, limits in thresholds.items():
        for share in shares:
            rng = np.random.default_rng([seed, int(round(share * 100)), size])
            hits = {key: 0 for key in measures}
            for _ in range(n_windows):
                values = measure_window(measures, ref_emb, emb[mix_window(group, rest, share, size, rng)])
                for key in measures:
                    hits[key] += values[key] > limits[key]
            rows[(size, share)] = {key: count / n_windows for key, count in hits.items()}
    return pd.DataFrame(rows).T.rename_axis(["Fenster", "Anteil"])


def false_alarm_rates(measures: dict, ref_emb: np.ndarray, pool_emb: np.ndarray, thresholds: dict[str, float],
                      window_size: int, n_windows: int = 300, seed: int = 2026) -> dict[str, float]:
    """Anteil frischer driftfreier Fenster, in denen ein Maß seine Schwelle überschreitet.

    Frisch heißt: anderer Seed als bei der Schwelle, die Fenster haben die
    Schwelle also nicht mitbestimmt. "irgendein Maß" zählt Fenster, in denen
    mindestens ein Maß alarmiert (so erlebt es die Rufbereitschaft).
    """
    rng = np.random.default_rng(seed)
    hits = {key: 0 for key in measures}
    any_hits = 0
    for _ in range(n_windows):
        values = measure_window(measures, ref_emb, pool_emb[rng.choice(len(pool_emb), window_size, replace=False)])
        alarms = {key: values[key] > thresholds[key] for key in measures}
        for key, alarm in alarms.items():
            hits[key] += alarm
        any_hits += any(alarms.values())
    return {**{key: hits[key] / n_windows for key in measures}, "irgendein Maß": any_hits / n_windows}


# ---------------------------------------------------------------- Wochenreihen (Übung 2)

WEEKLY_MEANS = {"accuracy": "correct", "mean_confidence": "confidence", "mean_top1_sim": "top1_sim"}


def weekly_table(measures: dict, ref_emb: np.ndarray, log: RequestLog, weekly: dict[str, list[np.ndarray]],
                 weekly_shares: dict[str, list[float]]) -> pd.DataFrame:
    """Eine Zeile je Reihe und Woche: Anteil, die Embedding-Maße und die Mittelwerte aus WEEKLY_MEANS.

    accuracy braucht Labels (im Betrieb erst Wochen später), die übrigen Spalten nicht.
    """
    rows = []
    for series, windows in weekly.items():
        for week, (share, idx) in enumerate(zip(weekly_shares[series], windows), start=1):
            rows.append({"series": series, "week": week, "share": round(float(share), 2),
                         **measure_window(measures, ref_emb, log.emb[idx]),
                         **{key: float(log.signals[column].to_numpy(float)[idx].mean()) for key, column in WEEKLY_MEANS.items()}})
    return pd.DataFrame(rows)


def alarm_weeks(weeks: pd.DataFrame, upper: dict[str, float], lower: dict[str, float]) -> pd.DataFrame:
    """Erste Alarmwoche je Signal (Zeilen) und Reihe (Spalten); None = kein Alarm.

    upper: Schwellen der Embedding-Maße (Alarm darüber), lower: untere Schwellen der Mittelwerte (Alarm darunter).
    """
    table = {series: {**{key: first_alarm(part[key], value) for key, value in upper.items()},
                      **{key: first_alarm(part[key], value, below=True) for key, value in lower.items()}}
             for series, part in weeks.groupby("series", sort=False)}
    return pd.DataFrame(table, dtype=object)


def week_review(lab: SimpleNamespace, weeks: pd.DataFrame, lower: dict[str, float], alarms: pd.DataFrame,
                upper: dict[str, float], false_alarm_rates_500: dict[str, float], window: int = WINDOW,
                seed: int = SEED, label_delay: int = 3) -> SimpleNamespace:
    """Kennzahlen zu den Wochenreihen, wenn die Labels eingetroffen sind; report enthält die Ausgabe als Text.

    Accuracy driftfreier Wochenfenster (Mittel, Streuung, Nachweisgrenze bei 50 % und 90 %), ruhige Wochen 1 und 2,
    Accuracy je Anliegen-Gruppe auf allen Testanfragen und die daraus folgende Erwartung für Woche 8 der
    Themenverschiebung, Label-Verzug, Fehlalarme in den ruhigen Wochen (Raten aus frischen driftfreien Fenstern).
    """
    z90 = 1.2816  # ein Verlust 1,28 Streuungen jenseits der Schwelle fällt in 90 % der Wochen auf
    log = lab.log

    def mean_of(reference, current):
        return current.mean()

    acc_values = log.signals["correct"].to_numpy(float)[log.pool]
    _, acc_null = drift.null_threshold(mean_of, acc_values, acc_values, window, seed=seed)
    acc_mean, acc_sd = float(acc_null.mean()), float(acc_null.std(ddof=1))
    limit_50 = acc_mean - lower["accuracy"]
    limit_90 = limit_50 + z90 * acc_sd
    quiet = weeks[weeks["week"] <= 2]
    clean = pd.concat([lab.ref_sig, log.signals.iloc[log.pool]], ignore_index=True)
    is_card = np.isin(np.r_[lab.ref["label"].to_numpy(), log.labels[log.pool]], [lab.intents.index(n) for n in lab.card_intents])
    acc_card, acc_other = float(clean["correct"][is_card].mean()), float(clean["correct"][~is_card].mean())
    topic = weeks[weeks["series"] == "Themenverschiebung"].assign(
        erwartet=lambda d: acc_mean - (d["share"] - lab.base_share) * (acc_other - acc_card))
    last = topic.iloc[-1]
    expected_loss = acc_mean - float(last["erwartet"])
    n_needed = ((3 + z90) * np.sqrt(acc_mean * (1 - acc_mean)) / expected_loss) ** 2
    quiet_rates = {key: float(false_alarm_rates_500[key]) for key in upper}
    for key, column in WEEKLY_MEANS.items():
        values = log.signals[column].to_numpy(float)[log.pool]
        _, fresh = drift.null_threshold(mean_of, values, values, window, n_boot=2000, seed=2026)
        quiet_rates[key] = float((fresh < lower[key]).mean())
    typo = weeks[weeks["series"] == "Tippfehler"].set_index("week")
    typo_week = alarms.loc["accuracy", "Tippfehler"]
    n_text = f"{round(n_needed, -2):,.0f}".replace(",", " ")
    lines = [
        f"Accuracy driftfreier Wochen: Mittel {acc_mean:.3f}, Streuung {acc_sd:.3f}, untere Schwelle {lower['accuracy']:.3f}",
        f"Nachweisgrenze: ein Verlust von {limit_50 * 100:.1f} Punkten fällt in der Hälfte der Wochen auf, "
        f"in 90 % erst einer ab {limit_90 * 100:.1f} Punkten",
        f"Ruhige Wochen (1 und 2, beide Reihen): {quiet['accuracy'].round(3).tolist()} | Abstand zum Mittel in Streuungen: "
        f"{((quiet['accuracy'] - acc_mean) / acc_sd).round(1).tolist()}",
        f"Accuracy auf allen Testanfragen: Karten-Anliegen {acc_card:.3f}, übrige {acc_other:.3f}",
        f"Themenverschiebung, Woche {int(last['week'])} ({last['share']:.0%} Karten-Anliegen): erwartet aus den Gruppen "
        f"{last['erwartet']:.3f}, gemessen {last['accuracy']:.3f}",
        f"Erwarteter Unterschied zum Mittel in Woche {int(last['week'])} (Punktschätzung): {expected_loss * 100:.1f} Punkte; "
        f"eine gelabelte Stichprobe, die ihn in 90 % der Fälle nachweist (Schwelle 3 Standardfehler), bräuchte rund {n_text} Anfragen",
    ]
    if not pd.isna(typo_week):
        lines.append(f"Tippfehler: Accuracy-Alarm in Woche {int(typo_week)}, mit {label_delay} Wochen Label-Verzug "
                     f"sichtbar in Woche {int(typo_week) + label_delay}")
    lines += [
        f"Tippfehler, Woche 1 (noch ohne Tippfehler): Domain-Classifier {typo.loc[1, 'domain_auc']:.4f} "
        f"gegen Schwelle {upper['domain_auc']:.4f}",
        f"Erwartete Fehlalarme in den 24 ruhigen Tests (6 Signale, Wochen 1 und 2, beide Reihen): {4 * sum(quiet_rates.values()):.2f}",
    ]
    return SimpleNamespace(acc_mean=acc_mean, acc_sd=acc_sd, limit_50=limit_50, limit_90=limit_90, quiet=quiet,
                           acc_card=acc_card, acc_other=acc_other, topic=topic, expected_loss=expected_loss,
                           n_needed=float(n_needed), quiet_rates=quiet_rates, report="\n".join(lines))


# ---------------------------------------------------------------- E-Mail-Kanal (Übung 1)

MAIL_GREETINGS = ("", "", "Hi,", "Hello,", "Hi there,", "Dear Sir or Madam,", "Good morning,")
MAIL_CLOSINGS = ("", "Thanks", "Thank you!", "Thanks, Anna", "Best regards,\nTom Miller", "Kind regards,\nJ. Schmidt", "Cheers, Ben")
MAIL_CLOSINGS_NAMED = {"Thanks, Anna", "Best regards,\nTom Miller", "Kind regards,\nJ. Schmidt", "Cheers, Ben"}
MAIL_SIGNATURES = ("", "", "Sent from my iPhone", "Sent from Outlook for Android", "Von meinem iPhone gesendet", "Get Outlook for iOS")
# Was eine einfache Vorverarbeitung sicher erkennt: Anrede-Zeilen und Geräte-Signaturen. Grußformeln mit Namen bleiben stehen.
_MAIL_BOILERPLATE = re.compile(r"^(hi|hello|hi there|dear sir or madam|good morning),?$|^(sent from|von meinem|get outlook)\b.*$",
                               re.I | re.M)


def _mail_parts(text: str, rng: np.random.Generator) -> tuple[str, str, str, str]:
    """Anrede, Text, Grußformel, Signatur (leere Teile fehlen später in der Mail)."""
    return str(rng.choice(MAIL_GREETINGS)), text, str(rng.choice(MAIL_CLOSINGS)), str(rng.choice(MAIL_SIGNATURES))


def _join_mail(parts) -> str:
    return "\n".join(part for part in parts if part)


def as_mail(text: str, rng: np.random.Generator) -> str:
    """Eine Anfrage, wie sie per Mail kommt: wechselnde Anrede, Grußformel und Signatur (jeweils auch ohne)."""
    return _join_mail(_mail_parts(text, rng))


def strip_mail(text: str) -> str:
    """Vorverarbeitung im Service: Anrede-Zeilen und Geräte-Signaturen entfernen, alles andere bleibt."""
    return "\n".join(line for line in _MAIL_BOILERPLATE.sub("", text).splitlines() if line.strip())


def mail_pilot(lab: SimpleNamespace, seed: int = SEED) -> pd.DataFrame:
    """Gelabelter Pilot: alle Pool-Anfragen einmal als Chat, als Mail und als Mail nach strip_mail.

    Dazu eine Kontrolle: nach strip_mail zusätzlich die Grußformeln mit Namen entfernt.
    Liefert je Variante Accuracy und mittlere Konfidenz. Embeddings aus labs/.cache/.
    """
    log = lab.log
    rng = np.random.default_rng([seed, 2])
    plain = [log.texts[i] for i in log.pool]
    labels = log.labels[log.pool]
    parts = [_mail_parts(text, rng) for text in plain]  # dieselben Ziehungen wie as_mail
    mails = [_join_mail(p) for p in parts]
    variants = {"Chat (ohne Mail-Rahmen)": (plain, log.emb[log.pool]),
                "Mail": (mails, cached_encode(lab.encoder, "minilm-l6", mails, "mail"))}
    stripped = [strip_mail(text) for text in mails]
    variants["Mail nach strip_mail"] = (stripped, cached_encode(lab.encoder, "minilm-l6", stripped, "mailstrip"))
    # Kontrolle: nach strip_mail zusätzlich die Grußformeln mit Namen entfernen
    control = [strip_mail(_join_mail((g, t, "" if c in MAIL_CLOSINGS_NAMED else c, s))) for g, t, c, s in parts]
    variants["Kontrolle: zusätzlich ohne Grußformel mit Namen"] = (
        control, cached_encode(lab.encoder, "minilm-l6", control, "mailctrl"))
    rows = {}
    for name, (texts, emb) in variants.items():
        sig = compute_signals(texts, emb, labels, lab.predict_proba, lab.index_emb, lab.index_labels)
        rows[name] = {"accuracy": float(sig["correct"].mean()), "mean_confidence": float(sig["confidence"].mean()),
                      "n": len(texts)}
    return pd.DataFrame(rows).T.astype({"n": int})


def sha256_file(path: str | Path) -> str:
    """SHA-256 einer Datei (zum Beispiel der Gewichte), stückweise gelesen."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- alles auf einmal (für die Übung)

def prepare(short_name: str = "minilm-l6", seed: int = SEED) -> SimpleNamespace:
    """Baut den Stand am Ende der Demo: Index, Klassifikator, Referenz, Log, Szenarien, Wochenreihen.

    Embeddings kommen aus labs/.cache/ (beim ersten Lauf werden sie gerechnet).
    """
    from .data import load_banking77
    from .models import load_encoder

    train, test, intents = load_banking77()
    encoder = load_encoder(short_name)
    index_emb = cached_encode(encoder, short_name, train["text"].tolist(), "index")
    index_labels = train["label"].to_numpy()
    predict_proba, classifier_info = load_classifier(index_emb, index_labels, len(intents))
    ref, pool = split_reference(test, seed=seed)
    ref_emb = cached_encode(encoder, short_name, ref["text"].tolist(), "ref")
    ref_sig = compute_signals(ref["text"].tolist(), ref_emb, ref["label"].to_numpy(), predict_proba, index_emb, index_labels)
    log = build_log(pool, encoder, predict_proba, index_emb, index_labels, short_name, seed)
    card_intents, card_pos, other_pos, base_share = topic_groups(log, intents)
    weekly, weekly_shares = weekly_windows(log, card_pos, other_pos, base_share, seed=seed + 1)
    gamma = drift.median_gamma(ref_emb, seed=seed)
    return SimpleNamespace(
        train=train, test=test, intents=intents, encoder=encoder, index_emb=index_emb, index_labels=index_labels,
        predict_proba=predict_proba, classifier_info=classifier_info, ref=ref, ref_emb=ref_emb, ref_sig=ref_sig,
        log=log, card_intents=card_intents, card_pos=card_pos, other_pos=other_pos, base_share=base_share,
        windows=scenario_windows(log, card_pos, other_pos, seed=seed), weekly=weekly, weekly_shares=weekly_shares,
        gamma=gamma, measures=embedding_measures(gamma, seed),
    )
