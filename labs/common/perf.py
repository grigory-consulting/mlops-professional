"""Messbausteine für Lab 5: Encoder-Varianten laden, Tempo messen, Kosten rechnen, Cache simulieren.

Drei Varianten desselben Encoders `minilm-l6` werden verglichen:

- `pytorch`: das Original (safetensors, PyTorch),
- `onnx`: derselbe Rechengraph als ONNX-Datei, ausgeführt von onnxruntime (float32),
- `onnx-int8`: die ONNX-Datei mit dynamisch auf int8 quantisierten Gewichten.

Jede Variante wird dreifach gemessen: Größe auf der Festplatte, Geschwindigkeit, Qualität
(nDCG@10 auf dem festen Eval-Set). Alle Zeitmessungen laufen mit fester Thread-Zahl
(`N_THREADS`), mit Warmlauf und als Median über Wiederholungen: Andere Programme
machen eine Messung nur langsamer, der Median hält einzelne Ausreißer heraus.
"""
from __future__ import annotations

import hashlib
import platform
import re
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .models import encode, model_path
from .paths import CACHE_DIR

# Vier Threads entsprechen einem kleinen Service-Container. Gilt für torch UND onnxruntime.
N_THREADS = 4
VARIANTS = ("pytorch", "onnx", "onnx-int8")


# ---------------------------------------------------------------------------
# Threads, onnxruntime, Quantisierung
# ---------------------------------------------------------------------------

def set_threads(n_threads: int = N_THREADS) -> int:
    """Legt die Thread-Zahl von PyTorch fest (onnxruntime: siehe ort_kwargs)."""
    import torch

    torch.set_num_threads(n_threads)
    return n_threads


def ort_kwargs(file_name: str | None = None, n_threads: int = N_THREADS) -> dict:
    """Einstellungen für onnxruntime: nur CPU, feste Thread-Zahl, optional eine bestimmte ONNX-Datei.

    Ohne festen Provider nimmt sentence-transformers den ersten, den onnxruntime
    meldet. Auf einem Mac ist das CoreML, und dort bricht dieses Modell beim
    Kodieren mit einem Laufzeitfehler ab.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = n_threads
    options.inter_op_num_threads = 1
    kwargs = {"provider": "CPUExecutionProvider", "session_options": options}
    if file_name:
        kwargs["file_name"] = file_name
    return kwargs


def quantization_target() -> str:
    """Quantisierungs-Konfiguration passend zum Befehlssatz der CPU.

    arm64 (Apple Silicon, ARM-Server) und avx512_vnni speichern Gewichte als int8.
    Ohne VNNI (nur AVX2) kann die Summe aus uint8 mal int8 überlaufen, deshalb
    nimmt die Konfiguration avx2 auch für die Gewichte uint8.
    """
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm64"
    cpuinfo = Path("/proc/cpuinfo")  # nur unter Linux lesbar
    if cpuinfo.exists() and "avx512_vnni" in cpuinfo.read_text():
        return "avx512_vnni"
    return "avx2"


def int8_file_name() -> str:
    """Relativer Pfad der quantisierten Datei im ONNX-Ordner, z. B. onnx/model_qint8_arm64.onnx."""
    return f"onnx/model_qint8_{quantization_target()}.onnx"


def export_onnx(out_dir: Path, force: bool = False) -> tuple[Path, Path]:
    """Exportiert minilm-l6 nach ONNX und quantisiert dynamisch auf int8 (nur falls nötig).

    Liefert (Pfad float32-Datei, Pfad int8-Datei). Liegen beide Dateien schon
    vor, passiert nichts. force=True löscht den Ordner und exportiert frisch.
    """
    from sentence_transformers import SentenceTransformer, export_dynamic_quantized_onnx_model

    out_dir = Path(out_dir)
    fp32, int8 = out_dir / "onnx" / "model.onnx", out_dir / int8_file_name()
    if force:
        shutil.rmtree(out_dir, ignore_errors=True)
    if not fp32.exists():
        exported = SentenceTransformer(model_path("minilm-l6"), backend="onnx", device="cpu", model_kwargs=ort_kwargs())
        exported.save_pretrained(str(out_dir))
    if not int8.exists():
        onnx_model = SentenceTransformer(str(out_dir), backend="onnx", device="cpu",
                                         model_kwargs=ort_kwargs("onnx/model.onnx"))
        target = quantization_target()
        export_dynamic_quantized_onnx_model(onnx_model, target, str(out_dir), file_suffix=f"qint8_{target}")
    return fp32, int8


# ---------------------------------------------------------------------------
# Varianten
# ---------------------------------------------------------------------------

REFERENCE_FILE = CACHE_DIR.parent / "reference" / "lab5_zielhardware.json"


def load_reference(path: Path = REFERENCE_FILE) -> dict | None:
    """Referenzmessung der Zielhardware (feste Datei im Kurspaket), ohne Datei None.

    Die Datei wird beim Neubau der Demo nicht überschrieben: Entscheidungen der
    Übung bleiben auf jedem Teilnehmer-Rechner dieselben.
    """
    import json

    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def file_mb(path) -> float:
    """Größe einer Datei auf der Festplatte in MB (10^6 Byte)."""
    return Path(path).stat().st_size / 1e6


def file_fingerprint(path, n: int = 8) -> str:
    """Kurzer SHA-256 einer Datei: zwei Exporte mit gleichem Fingerabdruck rechnen gleich."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


@dataclass
class Variant:
    """Eine Encoder-Variante: Modell, Gewichtsdatei auf der Festplatte und abgeleitete Version."""

    name: str
    model: object
    weights: Path
    version: str = field(default="")

    def __post_init__(self) -> None:
        self.weights = Path(self.weights)
        if not self.version:
            # Name plus Fingerabdruck der Gewichte: ändert sich die Datei, ändert sich die Version
            self.version = f"{self.name}-{file_fingerprint(self.weights)}"

    @property
    def size_mb(self) -> float:
        return file_mb(self.weights)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return encode(self.model, texts, batch_size)

    def __call__(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.encode(texts, batch_size)


def load_variant(name: str, onnx_dir: Path | None = None) -> Variant:
    """Lädt eine der drei Varianten von minilm-l6 (onnx_dir: Ordner aus export_onnx)."""
    from sentence_transformers import SentenceTransformer

    if name == "pytorch":
        model = SentenceTransformer(model_path("minilm-l6"), device="cpu")
        return Variant(name, model, Path(model_path("minilm-l6")) / "model.safetensors")
    if onnx_dir is None:
        raise ValueError("Für die ONNX-Varianten fehlt onnx_dir (Ordner aus export_onnx).")
    file_name = {"onnx": "onnx/model.onnx", "onnx-int8": int8_file_name()}[name]
    model = SentenceTransformer(str(onnx_dir), backend="onnx", device="cpu", model_kwargs=ort_kwargs(file_name))
    return Variant(name, model, Path(onnx_dir) / file_name)


def load_encoder_variant(short_name: str) -> Variant:
    """Ein anderes Embedding-Modell des Kurses (z. B. bge-small) als Variant, PyTorch auf CPU."""
    from .models import load_encoder

    return Variant(short_name, load_encoder(short_name), Path(model_path(short_name)) / "model.safetensors")


# ---------------------------------------------------------------------------
# Embeddings mit Datei-Cache
# ---------------------------------------------------------------------------

def texts_fingerprint(texts: list[str]) -> str:
    h = hashlib.sha256()
    for text in texts:
        h.update(text.encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()[:16]


def cached_embeddings(variant: Variant, texts: list[str], label: str, cache_dir: Path = CACHE_DIR) -> np.ndarray:
    """Embeddings aus labs/.cache/, sonst rechnen und ablegen.

    Der Dateiname enthält Modellversion (mit Fingerabdruck der Gewichte) und
    Fingerabdruck der Texte: Ein neuer Export oder andere Texte treffen nie eine
    alte Datei. Derselbe Gedanke wie beim Cache-Schlüssel in Demo 5.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"lab5_{variant.version}_{label}_{texts_fingerprint(texts)}.npy"
    if path.exists():
        return np.load(path)
    emb = variant.encode(texts, 64)
    np.save(path, emb)
    return emb


def evalset_embeddings(variant: Variant, ev) -> tuple[np.ndarray, np.ndarray]:
    """(Query-Embeddings, Korpus-Embeddings) des Eval-Sets, über den Datei-Cache."""
    return (cached_embeddings(variant, ev.queries, f"queries-{ev.fingerprint}"),
            cached_embeddings(variant, ev.corpus, f"corpus-{ev.fingerprint}"))


# ---------------------------------------------------------------------------
# Zeitmessung
# ---------------------------------------------------------------------------

EncodeFn = Callable[[list[str], int], np.ndarray]


def single_latency(encode_fn: EncodeFn, texts: list[str], n: int = 200, n_warmup: int = 20) -> dict:
    """Latenz einzelner Anfragen (Batchgröße 1) nach Warmlauf: Median und p95 in ms."""
    for text in texts[:n_warmup]:
        encode_fn([text], 1)
    latencies = []
    for text in texts[:n]:
        start = time.perf_counter()
        encode_fn([text], 1)
        latencies.append(1000 * (time.perf_counter() - start))
    return {"latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p95_ms": float(np.percentile(latencies, 95))}


def throughput(encode_fn: EncodeFn, texts: list[str], batch_size: int = 64, repeats: int = 3) -> float:
    """Texte je Sekunde, wenn alle texts mit batch_size kodiert werden (Median über repeats).

    encode von sentence-transformers sortiert die Texte vorher nach Länge: Das ist
    der Durchsatz beim Indexieren des Korpus, nicht der des Anfragepfads.
    """
    encode_fn(texts[: 2 * batch_size], batch_size)  # Warmlauf
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        encode_fn(texts, batch_size)
        durations.append(time.perf_counter() - start)
    return float(len(texts) / np.median(durations))


def benchmark(encode_fn: EncodeFn, texts: list[str], batch_size: int = 64) -> dict:
    """Einzel-Latenz (Median, p95) und Durchsatz im Batch, für jede Variante gleich gemessen."""
    return {**single_latency(encode_fn, texts), "throughput_per_s": throughput(encode_fn, texts, batch_size)}


def batch_curve(encode_fn: EncodeFn, texts: list[str], batch_sizes=(1, 2, 4, 8, 16, 32, 64),
                min_batches: int = 10, min_texts: int = 128) -> list[dict]:
    """Latenz je Batch (Median, p95) und Durchsatz je Batchgröße, Texte in Eingangsreihenfolge.

    Jeder Batch besteht aus aufeinanderfolgenden Texten, wie sie im Service
    eintreffen: gemischte Längen, also Padding auf den längsten Text im Batch.
    Je Batchgröße mindestens min_batches Batches und min_texts Texte (zyklisch
    über texts), damit auch kleine Batchgrößen genug Messpunkte haben.
    """
    curve = []
    doubled = list(texts) + list(texts)  # erlaubt Batches über das Ende der Liste hinaus
    for b in batch_sizes:
        n_batches = max(min_batches, -(-min_texts // b))
        encode_fn(texts[:b], b)  # Warmlauf mit dieser Batchgröße
        durations = []
        for i in range(n_batches):
            start_idx = (i * b) % len(texts)
            batch = doubled[start_idx : start_idx + b]
            start = time.perf_counter()
            encode_fn(batch, b)
            durations.append(time.perf_counter() - start)
        p50 = float(np.median(durations))
        curve.append({
            "batch_size": int(b),
            "latency_p50_ms": 1000 * p50,
            "latency_p95_ms": 1000 * float(np.percentile(durations, 95)),
            "throughput_per_s": b / p50,
        })
    return curve


def latency_by_tokens(encode_fn: EncodeFn, texts: list[str], tokenizer, n: int = 200, n_warmup: int = 20) -> dict:
    """Einzel-Latenz gegen Token-Zahl: Gerade ms = Sockel + Steigung * Tokens, dazu die Korrelation.

    Liefert außerdem den Wert der Geraden bei der mittleren (Median) und der größten
    Token-Zahl: Sockel und Steigung zeigen zusammen, wie stark die Länge zählt.
    """
    tokens = np.array([len(ids) for ids in tokenizer(texts[:n])["input_ids"]])
    for text in texts[:n_warmup]:
        encode_fn([text], 1)
    latencies = []
    for text in texts[:n]:
        start = time.perf_counter()
        encode_fn([text], 1)
        latencies.append(1000 * (time.perf_counter() - start))
    slope, intercept = np.polyfit(tokens, latencies, 1)
    t50, tmax = float(np.median(tokens)), int(tokens.max())
    return {"base_ms": float(intercept), "ms_per_token": float(slope),
            "corr": float(np.corrcoef(tokens, latencies)[0, 1]),
            "tokens_p50": t50, "tokens_max": tmax,
            "fit_ms_at_p50": float(intercept + slope * t50), "fit_ms_at_max": float(intercept + slope * tmax)}


def torch_blas() -> str:
    """BLAS-Bibliothek, mit der PyTorch gebaut ist (z. B. accelerate auf dem Mac, mkl oder openblas auf x86)."""
    import torch

    match = re.search(r"BLAS_INFO=(\w+)", torch.__config__.show())
    return match.group(1) if match else "unbekannt"


def measure_timings(variants: dict, texts: list[str], batch_sizes=(1, 2, 4, 8), curve_texts: int = 512) -> tuple[dict, dict]:
    """Kurze Tempo-Messung auf diesem Rechner, nach demselben Verfahren wie die Referenzmessung.

    Liefert (je Variante Einzel-Latenz und Durchsatz, je Variante Batch-Kurve).
    Die Kurve deckt nur kleine Batchgrößen ab, damit die Messung auch auf
    langsamen Laptops kurz bleibt.
    """
    timings, curves = {}, {}
    for name, variant in variants.items():
        timings[name] = benchmark(variant, texts)
        curves[name] = batch_curve(variant, texts[:curve_texts], batch_sizes)
    return timings, curves


def padding_factor(tokenizer, texts: list[str], batch_sizes=(1, 2, 4, 8, 16, 32, 64)) -> dict[int, float]:
    """Gerechnete Tokens (mit Auffüllen auf den längsten Text im Batch) je echtem Token."""
    counts = np.array([len(ids) for ids in tokenizer(texts)["input_ids"]])
    factors = {}
    for b in batch_sizes:
        chunks = [counts[i : i + b] for i in range(0, len(counts), b)]
        factors[int(b)] = float(sum(c.max() * len(c) for c in chunks) / counts.sum())
    return factors


# ---------------------------------------------------------------------------
# Kosten
# ---------------------------------------------------------------------------

def cost_per_1000(throughput_per_s: float, price_per_cpu_hour: float,
                  n_cpus: int = N_THREADS, utilization: float = 1.0) -> float:
    """Kosten je 1 000 Anfragen = Preis je Stunde für n_cpus / (Anfragen je Stunde / 1 000).

    Der Durchsatz wurde mit n_cpus Kernen gemessen, also kostet eine Stunde
    n_cpus CPU-Stunden. utilization < 1: Der Service ist nicht voll ausgelastet,
    jede Anfrage trägt anteilig die Leerlaufzeit mit.
    """
    requests_per_hour = throughput_per_s * 3600 * utilization
    return price_per_cpu_hour * n_cpus / (requests_per_hour / 1000)


# ---------------------------------------------------------------------------
# Embedding-Cache
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Gleiche Anfrage, gleiche Schreibweise: Rand-Leerraum weg, Leerraum zusammenfassen, Kleinbuchstaben.

    Kleinschreibung ist nur bei einem uncased-Modell (wie minilm-l6) verlustfrei.
    """
    return re.sub(r"\s+", " ", text.strip()).lower()


def cache_key(model_version: str, text: str) -> str:
    """SHA-256 über Modellversion und normalisierten Text."""
    payload = f"{model_version}\x1f{normalize_text(text)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LRUCache:
    """Hält höchstens capacity Einträge und verdrängt den am längsten nicht genutzten."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.store: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.store:
            self.store.move_to_end(key)  # als zuletzt genutzt markieren
            self.hits += 1
            return self.store[key]
        self.misses += 1
        return None

    def put(self, key, value) -> None:
        self.store[key] = value
        self.store.move_to_end(key)
        if len(self.store) > self.capacity:
            self.store.popitem(last=False)  # ältesten Eintrag verdrängen

    @property
    def hit_rate(self) -> float:
        return self.hits / max(1, self.hits + self.misses)


def make_stream(pool: list[str], n_requests: int = 5000, zipf_s: float = 1.0,
                vary_share: float = 0.3, seed: int = 1) -> tuple[list[str], np.ndarray]:
    """Anfragestrom: Zipf-verteilte Häufigkeit (wenige Fragen sehr oft, viele selten).

    vary_share der Anfragen ist anders getippt (Großbuchstaben oder Leerraum).
    Liefert (Texte, Index in pool je Anfrage). Fester Seed: auf jedem Rechner gleich.
    """
    rng = np.random.default_rng(seed)
    popularity = rng.permutation(len(pool))  # legt fest, welche Anfrage wie beliebt ist
    weights = 1.0 / np.arange(1, len(pool) + 1) ** zipf_s
    draws = popularity[rng.choice(len(pool), size=n_requests, p=weights / weights.sum())]
    r = rng.random(n_requests)
    texts = [pool[i].upper() if x < vary_share / 2 else (f"  {pool[i]}  " if x < vary_share else pool[i])
             for i, x in zip(draws, r)]
    return texts, draws


def simulate_hit_rate(keys, capacity: int) -> float:
    """Trefferquote eines LRU-Caches für eine Folge von Schlüsseln (ohne Encoder)."""
    cache = LRUCache(capacity)
    for key in keys:
        if cache.get(key) is None:
            cache.put(key, True)
    return cache.hit_rate


def zipf_sweep(pool: list[str], model_version: str, exponents, n_requests: int = 5000, seed: int = 1) -> dict:
    """Trefferquoten je Zipf-Exponent des simulierten Stroms (ohne Encoder, nur Schlüssel).

    Je Exponent: höchstens erreichbar (Anteil Wiederholungen), LRU 100 und LRU 1 000
    mit normalisiertem Schlüssel, LRU 1 000 mit rohem Text im Schlüssel.
    """
    sweep = {}
    for zipf_s in exponents:
        texts, _ = make_stream(pool, n_requests=n_requests, zipf_s=zipf_s, seed=seed)
        k_norm = [cache_key(model_version, t) for t in texts]
        k_raw = [f"{model_version}|{t}" for t in texts]  # roher Text im Schlüssel
        sweep[zipf_s] = {"bound": 1 - len(set(k_norm)) / len(k_norm),
                         "lru100": simulate_hit_rate(k_norm, 100), "lru1000": simulate_hit_rate(k_norm, 1000),
                         "lru1000_raw": simulate_hit_rate(k_raw, 1000)}
    return sweep


# ---------------------------------------------------------------------------
# Plots (Kursstil: vorher common.plotstyle.use_course_style() aufrufen)
# ---------------------------------------------------------------------------

LABELS = {"pytorch": "PyTorch", "onnx": "ONNX float32", "onnx-int8": "ONNX int8"}


def _de(value: float, fmt: str) -> str:
    return fmt.format(value).replace(".", ",")


def plot_variants(results: dict, title: str):
    """Vier Balkenpanels je Variante: Größe, Einzel-Latenz, Durchsatz, nDCG@10."""
    import matplotlib.pyplot as plt

    from .plotstyle import ACCENT, INK, WARN

    names = list(results)
    colors = [INK, ACCENT, WARN][: len(names)]
    panels = [
        ("size_mb", "Größe auf der Festplatte (MB)", "{:.0f}"),
        ("latency_p50_ms", "Latenz Einzelanfrage, Median (ms)", "{:.1f}"),
        ("throughput_per_s", "Durchsatz Batch 64, sortiert (Texte/s)", "{:.0f}"),
        ("ndcg_at_10", "nDCG@10", "{:.3f}"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (key, panel_title, fmt) in zip(axes, panels):
        values = [results[n][key] for n in names]
        bars = ax.bar([LABELS.get(n, n) for n in names], values, color=colors, width=0.6)
        ax.bar_label(bars, labels=[_de(v, fmt) for v in values], padding=2, color=INK)
        ax.set_title(panel_title, fontsize=11)
        ax.set_ylim(0, max(values) * 1.15)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=10)
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    return fig


def plot_batch_curves(curves: dict, title: str, highlight: tuple[str, int] | None = None):
    """Durchsatz und p95-Latenz je Batch über der Batchgröße; highlight=(Variante, Batchgröße) markiert einen Punkt."""
    import matplotlib.pyplot as plt

    from .plotstyle import ACCENT, INK, WARN

    fig, (ax_tp, ax_lat) = plt.subplots(1, 2, figsize=(12, 4.2))
    for (name, curve), color, marker in zip(curves.items(), [INK, ACCENT, WARN], ["o", "s", "^"]):
        sizes = [c["batch_size"] for c in curve]
        emphasis = highlight is None or highlight[0] == name
        style = {"color": color, "marker": marker, "label": LABELS.get(name, name),
                 "alpha": 1.0 if emphasis else 0.35, "linewidth": 2.2 if emphasis else 1.2}
        ax_tp.plot(sizes, [c["throughput_per_s"] for c in curve], **style)
        ax_lat.plot(sizes, [c["latency_p95_ms"] for c in curve], **style)
        if highlight and highlight[0] == name:
            point = next(c for c in curve if c["batch_size"] == highlight[1])
            for ax, key in ((ax_tp, "throughput_per_s"), (ax_lat, "latency_p95_ms")):
                ax.scatter([point["batch_size"]], [point[key]], s=180, facecolors="none",
                           edgecolors=color, linewidths=2.5, zorder=5)
    for ax in (ax_tp, ax_lat):
        ax.set_xscale("log", base=2)
        sizes = [c["batch_size"] for c in next(iter(curves.values()))]
        ax.set_xticks(sizes, [str(b) for b in sizes])
        ax.set_xlabel("Batchgröße (Texte gemischt, wie im Anfragepfad)")
    ax_tp.set_ylim(bottom=0)
    ax_tp.set_ylabel("Texte je Sekunde")
    ax_tp.set_title("Durchsatz", fontsize=11)
    ax_lat.set_yscale("log")
    ax_lat.set_ylabel("Millisekunden je Batch, p95 (log)")
    ax_lat.set_title("Latenz je Batch (p95)", fontsize=11)
    ax_lat.legend(frameon=False, loc="upper left")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def plot_hit_rates(sweep: dict, highlight: float | None = None):
    """Trefferquote gegen Zipf-Exponent (Ergebnis von zipf_sweep); highlight markiert einen Exponenten."""
    import matplotlib.pyplot as plt

    from .plotstyle import ACCENT, INK, WARN

    exponents = list(sweep)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for key, color, style, label in (("bound", INK, ":", "höchstens erreichbar"),
                                      ("lru1000", ACCENT, "-", "LRU 1 000, normalisiert"),
                                      ("lru1000_raw", WARN, "-", "LRU 1 000, roher Text"),
                                      ("lru100", ACCENT, "--", "LRU 100, normalisiert")):
        ax.plot(exponents, [sweep[e][key] for e in exponents], color=color, linestyle=style, marker="o", label=label)
    if highlight is not None:
        ax.axvline(highlight, color=INK, linestyle="--", linewidth=1)
    ax.set(xlabel="Zipf-Exponent des simulierten Stroms (Annahme)", ylabel="Trefferquote", ylim=(0, 1),
           title="Trefferquote gegen Wiederholungsrate des Traffics")
    ax.legend(frameon=False, loc="lower right")
    return fig
