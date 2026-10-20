"""Bausteine für Lab 3 (Demo und Übungen): Index bauen, Service konfigurieren, starten, prüfen, messen.

Der Service selbst liegt unverändert in `labs/lab3_service/`. Dieses Modul
liefert nur, was Notebooks rund um den Service brauchen:

- Batch-Pfad: `build_index`, `ensure_index` (Index mit meta.json, Fingerabdruck)
- Konfiguration: `choose_classifier`, `service_settings`, `service_env`, `service_command`
- Service im Prozess: `inprocess_client` (TestClient, mit oder ohne Startprüfung), `make_engine`
- Service als Prozess: `running_service`, `running_services` (uvicorn, sicher beendet),
  `process_memory_mb` (Speicher eines laufenden Service-Prozesses)
- Stichproben aus dem Index: `probe_positions`, `probe_cosines` (Vergleich mit den gespeicherten Vektoren),
  `ProbeScan` (Vorarbeit für den Regler in Demo 4)
- Blue-Green: `write_deployment`, `switch_deployment`, `Router` (liest deployment.json bei jeder Anfrage),
  `BackgroundStream` (Anfragestrom in Hintergrund-Threads), `print_stream` (Farbe und Paar je Anfrage)
- Last: `burst` (parallele Clients ohne Pause gegen einen oder mehrere Prozesse)
- Qualität: `engine_quality` (ganzes Eval-Set im Batch), `api_quality`, `api_ndcg_per_query` (über /similar),
  `bootstrap_mean` (Intervall eines Mittelwerts)
- Anzeige: `show_response`, `plot_pair_matrix`; Container: `docker`, `container_env`
- Übung 3: `STRATEGIES`, `rollout_feedback` (Rückmeldung nach der eigenen Entscheidung)
"""
from __future__ import annotations

import atexit
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from .evalset import EvalSet
from .metrics import retrieval_metrics
from .models import encode, load_encoder
from .paths import ARTIFACTS_DIR, CACHE_DIR, LABS_DIR, MODELS_DIR

LAB3_DIR = ARTIFACTS_DIR / "lab3"
# Klassifikator aus Lab 2; fehlt er, trainiert der Ersatzweg eine logistische Regression
CLASSIFIER_CANDIDATES = (
    ("lab2", ARTIFACTS_DIR / "lab2" / "classifier"),
)
# Diese Umgebungsvariablen liest lab3_service.engine.Settings
SERVICE_VARS = ("LAB_BASE_DIR", "ENCODER_PATH", "ENCODER_NAME", "INDEX_DIR", "CLASSIFIER_PATH")
RUN_DOCKER = os.environ.get("RUN_DOCKER") == "1"

_PROCESSES: list[subprocess.Popen] = []
_BY_URL: dict[str, subprocess.Popen] = {}
_CLIENTS: list = []


# --------------------------------------------------------------------------
# Batch-Pfad: Index bauen
# --------------------------------------------------------------------------
def _cache_file(model_name: str, evalset: EvalSet) -> Path:
    return CACHE_DIR / f"lab3_{model_name}_corpus_{evalset.fingerprint}.npy"


def build_index(model_name: str, encoder, evalset: EvalSet, corpus_intents: list[str], data_hash: str,
                index_dir: Path) -> tuple[dict, np.ndarray, float]:
    """Bettet den Korpus ein und schreibt den Index. Liefert (Metadaten, Embeddings, Sekunden).

    Derselbe Aufruf `encode` (normiert, batch_size 64) wie im Service: sonst
    entstünde schon hier ein Skew zwischen Index und Anfragepfad.
    """
    from lab3_service.engine import write_index

    start = time.perf_counter()
    corpus_emb = encode(encoder, evalset.corpus)
    seconds = time.perf_counter() - start
    meta = write_index(index_dir, corpus_emb, evalset.corpus, corpus_intents, model_name=model_name,
                       data_hash=data_hash, evalset_fingerprint=evalset.fingerprint)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(_cache_file(model_name, evalset), corpus_emb)
    return meta, corpus_emb, seconds


def ensure_index(model_name: str, evalset: EvalSet, corpus_intents: list[str], data_hash: str, out_dir: Path,
                 search_dirs: tuple[Path, ...] = (LAB3_DIR,), encoder=None) -> tuple[Path, str]:
    """Nimmt einen passenden Index aus search_dirs, sonst baut es ihn nach out_dir.

    Passend heißt: Fingerabdruck stimmt mit den Dateien überein, Modellname,
    Daten-Hash und Eval-Set sind dieselben. Liefert (Ordner, Herkunft).
    """
    from lab3_service.engine import load_index

    for folder in search_dirs:
        index_dir = Path(folder) / f"index_{model_name}"
        if not (index_dir / "meta.json").exists():
            continue
        try:
            meta = load_index(index_dir).meta
        except Exception:
            continue
        if (meta["model_name"], meta["data_hash"], meta["evalset_fingerprint"]) == (model_name, data_hash, evalset.fingerprint):
            return index_dir, f"vorhanden ({index_dir.relative_to(LABS_DIR)})"
    index_dir = Path(out_dir) / f"index_{model_name}"
    build_index(model_name, encoder or load_encoder(model_name), evalset, corpus_intents, data_hash, index_dir)
    return index_dir, f"neu gebaut ({index_dir.relative_to(LABS_DIR)})"


def corpus_embeddings(model_name: str, evalset: EvalSet) -> np.ndarray:
    """Korpus-Embeddings aus dem Cache von build_index, sonst neu gerechnet."""
    path = _cache_file(model_name, evalset)
    if path.exists():
        return np.load(path, allow_pickle=False)
    emb = encode(load_encoder(model_name), evalset.corpus)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


# --------------------------------------------------------------------------
# Konfiguration des Service
# --------------------------------------------------------------------------
@dataclass
class ClassifierChoice:
    """Welcher Klassifikator in den Service kommt und wie der Service ihn findet.

    label     lab2 oder fallback-logreg
    path      Ordner des fein abgestimmten BERT (wird als CLASSIFIER_PATH gesetzt) oder None
    base_dir  LAB_BASE_DIR des Service; beim Ersatzweg liegt darunter artifacts/lab3/logreg_<encoder>.npz
    """

    label: str
    path: Path | None
    base_dir: Path
    evalset: EvalSet = field(repr=False)
    intents: list[str] = field(repr=False)

    def ensure_head(self, encoder_name: str) -> None:
        """Ersatzweg: logistische Regression auf den Embeddings dieses Encoders trainieren (einmal)."""
        if self.path is not None:
            return
        from sklearn.linear_model import LogisticRegression

        from lab3_service.engine import write_logreg

        target = self.base_dir / "artifacts" / "lab3" / f"logreg_{encoder_name}.npz"
        if target.exists():
            return
        emb = corpus_embeddings(encoder_name, self.evalset)
        model = LogisticRegression(C=10, max_iter=200).fit(emb, self.evalset.corpus_labels)
        write_logreg(target, model.coef_, model.intercept_, self.intents, encoder_name=encoder_name)

    def describe(self) -> str:
        if self.path is None:
            return f"{self.label}: logistische Regression je Encoder unter {self.base_dir.relative_to(LABS_DIR)}"
        return f"{self.label}: {self.path.relative_to(LABS_DIR)} (CLASSIFIER_PATH)"


def choose_classifier(out_dir: Path, evalset: EvalSet, intents: list[str]) -> ClassifierChoice:
    """Reihenfolge: artifacts/lab2/classifier, sonst Ersatzweg LogReg."""
    for label, folder in CLASSIFIER_CANDIDATES:
        weights = (folder / "model.safetensors").exists() or (folder / "pytorch_model.bin").exists()
        if (folder / "config.json").exists() and weights:
            return ClassifierChoice(label, folder, LABS_DIR, evalset, intents)
    return ClassifierChoice("fallback-logreg", None, Path(out_dir) / "service_base", evalset, intents)


def service_settings(encoder: str, index_dir: Path, classifier: ClassifierChoice, encoder_name: str | None = None):
    """Settings des Service für ein Paar aus Encoder (Ordner unter models/) und Index.

    encoder_name überschreibt den Namen, den der Service für den Encoder meldet
    (zum Nachstellen einer falschen Beschriftung).
    """
    from lab3_service.engine import Settings

    classifier.ensure_head(encoder_name or encoder)
    return Settings(
        base_dir=classifier.base_dir,
        encoder_path=MODELS_DIR / encoder,
        encoder_name=encoder_name or encoder,
        encoder_source="env",
        index_dir=Path(index_dir),
        classifier_path=classifier.path,
    )


def service_env(settings) -> dict[str, str]:
    """Umgebungsvariablen, aus denen Settings.from_env dieselben Settings liest."""
    env = {
        "LAB_BASE_DIR": str(settings.base_dir),
        "ENCODER_PATH": str(settings.encoder_path),
        "ENCODER_NAME": settings.encoder_name,
        "INDEX_DIR": str(settings.index_dir),
    }
    if settings.classifier_path is not None:
        env["CLASSIFIER_PATH"] = str(settings.classifier_path)
    return env


def _relative(value: str) -> str:
    try:
        rel = Path(value).relative_to(LABS_DIR)
    except ValueError:
        return value
    return str(rel) if str(rel) != "." else "."


def service_command(settings, port: int = 8000) -> str:
    """Startbefehl für das Terminal (im Ordner labs/), mit relativen Pfaden."""
    env = {k: _relative(v) for k, v in service_env(settings).items()}
    if env.get("LAB_BASE_DIR") == ".":
        env.pop("LAB_BASE_DIR")  # Standard des Service: labs/
    if env.get("ENCODER_NAME") == Path(env["ENCODER_PATH"]).name:
        env.pop("ENCODER_NAME")
    prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    return f"{prefix} uvicorn lab3_service.app:app --port {port}"


# --------------------------------------------------------------------------
# Service im Prozess (TestClient)
# --------------------------------------------------------------------------
_CLASSIFIERS: dict = {}


def make_engine(settings, encoder=None, index=None):
    """Engine ohne Startprüfung aus vorhandenen Teilen (spart das erneute Laden der Modelle).

    Der Klassifikator wird je Konfiguration nur einmal geladen und dann wiederverwendet.
    """
    from sentence_transformers import SentenceTransformer

    from lab3_service.engine import Engine, load_classifier, load_index

    encoder = encoder or SentenceTransformer(str(settings.encoder_path), device="cpu")
    engine = Engine(settings, encoder, index or load_index(settings.index_dir), classifier=None)
    key = (settings.classifier_path, settings.base_dir, settings.encoder_name)
    if key not in _CLASSIFIERS:
        _CLASSIFIERS[key] = load_classifier(settings)
    engine.classifier = _CLASSIFIERS[key]
    return engine


def inprocess_client(settings, check: bool = True, engine=None):
    """TestClient für den Service aus lab3_service.

    check=True: echter Start über lifespan, einschließlich Prüfung von Encoder
    und Index (wirft IndexMismatchError, wenn sie nicht zusammenpassen).
    check=False: die Engine wird ohne Prüfung geladen (oder übergeben) und
    eingesetzt, wie bei einem Service, der beim Start nichts prüft.
    """
    from fastapi.testclient import TestClient

    from lab3_service.app import create_app
    from lab3_service.engine import load_engine

    app = create_app(settings)
    client = TestClient(app)
    if check and engine is None:
        client.__enter__()  # löst lifespan aus: Modelle laden, Paar prüfen
        _CLIENTS.append(client)
    else:
        app.state.engine = engine if engine is not None else load_engine(settings, check=False)
    return client


def _shorten(value, limit: int = 4):
    if isinstance(value, list):
        short = [_shorten(v, limit) for v in value[:limit]]
        return short + [f"... ({len(value)} Werte)"] if len(value) > limit else short
    if isinstance(value, dict):
        return {k: _shorten(v, limit) for k, v in value.items()}
    return round(value, 3) if isinstance(value, float) else value


def show_response(response, keys: tuple[str, ...] | None = None) -> None:
    """Zeigt Methode, Pfad, Statuscode und die (gekürzte) Antwort."""
    try:
        body = response.json()
    except ValueError:
        body = response.text[:300]
    if isinstance(body, dict) and keys:
        body = {k: body[k] for k in keys if k in body}
    if response.status_code == 422 and isinstance(body, dict):
        body = [{"loc": d["loc"], "msg": d["msg"]} for d in body.get("detail", [])]
    req = response.request
    print(f"{req.method} {req.url.path} -> {response.status_code}")
    print("  " + json.dumps(_shorten(body), ensure_ascii=False))


# --------------------------------------------------------------------------
# Service als eigener Prozess (uvicorn)
# --------------------------------------------------------------------------
class ServiceStartError(RuntimeError):
    """Der Service-Prozess ist nicht bereit geworden (Abbruch oder Zeitüberschreitung)."""


def free_port() -> int:
    """Freier Port vom Betriebssystem, nie 5000 (dort läuft ggf. MLflow)."""
    while True:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port != 5000:
            return port


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def stop_all() -> None:
    """Beendet alle Service-Prozesse und TestClients, die dieses Modul gestartet hat."""
    for proc in _PROCESSES:
        _stop(proc)
    while _CLIENTS:
        try:
            _CLIENTS.pop().__exit__(None, None, None)
        except Exception:
            pass


atexit.register(stop_all)


def running_count() -> int:
    """Anzahl der Service-Prozesse aus diesem Modul, die noch laufen."""
    return sum(p.poll() is None for p in _PROCESSES)


def wait_for_health(base_url: str, timeout: float = 120, alive=lambda: True) -> float:
    """Wartet, bis /health mit 200 antwortet. Liefert die Wartezeit in Sekunden."""
    import httpx

    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        if not alive():
            raise ServiceStartError("Prozess ist beim Start abgebrochen.")
        try:
            if httpx.get(base_url + "/health", timeout=2).status_code == 200:
                return time.perf_counter() - start
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise ServiceStartError(f"{base_url}/health antwortet nach {timeout:.0f} s nicht.")


@contextmanager
def running_services(deployments: dict, log_dir: Path, timeout: float = 120,
                     threads: int | None = None) -> Iterator[dict[str, str]]:
    """Startet je Eintrag {Name: Settings} einen uvicorn-Prozess, alle gleichzeitig.

    Liefert {Name: Basis-URL}. Beim Verlassen des with-Blocks (auch nach einem
    Fehler oder Abbruch) werden alle Prozesse beendet. threads begrenzt die
    Rechen-Threads von PyTorch je Prozess (wie die CPU-Grenze eines Containers);
    ohne Angabe nimmt PyTorch alle Kerne.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    started: dict[str, tuple[subprocess.Popen, str, Path]] = {}
    try:
        for name, settings in deployments.items():
            port = free_port()
            env = {k: v for k, v in os.environ.items() if k not in SERVICE_VARS} | service_env(settings)
            if threads is not None:
                env |= {"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
            log_path = log_dir / f"service_{name}.log"
            with open(log_path, "w") as log_file:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "uvicorn", "lab3_service.app:app", "--host", "127.0.0.1", "--port", str(port)],
                    cwd=LABS_DIR, env=env, stdout=log_file, stderr=subprocess.STDOUT,
                )
            _PROCESSES.append(proc)
            _BY_URL[f"http://127.0.0.1:{port}"] = proc
            started[name] = (proc, f"http://127.0.0.1:{port}", log_path)
        urls = {}
        for name, (proc, url, log_path) in started.items():
            try:
                wait_for_health(url, timeout, alive=lambda p=proc: p.poll() is None)
            except ServiceStartError as err:
                _stop(proc)
                tail = "\n".join(log_path.read_text().splitlines()[-3:])
                raise ServiceStartError(f"Service '{name}' startet nicht (Exit-Code {proc.returncode}): {err}\n{tail}") from None
            urls[name] = url
        yield urls
    finally:
        for proc, _, _ in started.values():
            _stop(proc)


@contextmanager
def running_service(name: str, settings, log_dir: Path, timeout: float = 120,
                    threads: int | None = None) -> Iterator[str]:
    """Wie running_services, für einen einzelnen Service. Liefert die Basis-URL."""
    with running_services({name: settings}, log_dir, timeout, threads) as urls:
        yield urls[name]


def process_memory_mb(base_url: str) -> float | None:
    """Belegter Arbeitsspeicher (RSS) des Service-Prozesses hinter base_url in MB, sonst None.

    Zählt die Kindprozesse mit. Unter Windows ist `python.exe` eines venv nur ein
    Umleiter, der den eigentlichen Interpreter als Kindprozess startet; ohne die
    Kinder käme dort nur der Speicher des Umleiters heraus. Auf macOS und Linux
    hat der Service-Prozess keine Kinder, die Summe ist dieselbe Zahl.
    """
    proc = _BY_URL.get(base_url)
    if proc is None or proc.poll() is not None:
        return None
    try:
        import psutil
    except ImportError:
        return None
    try:
        root = psutil.Process(proc.pid)
        family = [root, *root.children(recursive=True)]
    except psutil.Error:
        return None
    total = 0
    for p in family:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass  # Kind ist inzwischen beendet
    return total / 1e6


def wait_until_interrupted() -> None:
    """Blockiert, bis die Zelle gestoppt wird (Knopf Stop bzw. Interrupt)."""
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Zelle gestoppt, Service wird beendet.")


# --------------------------------------------------------------------------
# Stichproben aus dem Index nachrechnen
# --------------------------------------------------------------------------
def probe_positions(index, tokenizer, n: int, how: str = "erste") -> list[int]:
    """Positionen der Stichprobe im Index: die ersten n Einträge oder die n längsten (in Tokens).

    Der Service nimmt beim Start die ersten Einträge (PROBE_SIZE).
    """
    if how == "erste":
        return list(range(n))
    lengths = np.array([len(ids) for ids in tokenizer(index.texts)["input_ids"]])
    return [int(i) for i in np.argsort(-lengths, kind="stable")[:n]]


def probe_cosines(vectors: np.ndarray, index, positions: list[int]) -> np.ndarray:
    """Cosine-Similarity je Stichprobentext zwischen neu eingebettetem und gespeichertem Vektor."""
    return np.sum(vectors * index.embeddings[positions], axis=1)


class ProbeScan:
    """Vorarbeit für den Regler in Demo 4: Anfragepfade gegen Stichproben aus dem Index.

    Für jeden Anfragepfad in variants (Name -> Argumente für encode_variant)
    bettet es zwei Stichproben von je n_max Texten aus dem Index neu ein (die
    ersten und die längsten, siehe probe_positions) und misst nDCG@10 auf allen
    Queries. show(n) zeigt das Ergebnis für die ersten n Texte jeder Stichprobe.
    """

    SAMPLES = ("erste", "längste")

    def __init__(self, encoder, index, evalset: EvalSet, variants: dict[str, dict], n_max: int = 64):
        from lab3_service.engine import PROBE_MIN_COSINE, PROBE_SIZE

        from .encoder_eval import encode_variant

        self.variants, self.threshold, self.probe_size = variants, PROBE_MIN_COSINE, PROBE_SIZE
        positions = {how: probe_positions(index, encoder.tokenizer, n_max, how) for how in self.SAMPLES}
        self.tokens = {how: [len(encoder.tokenizer(index.texts[i])["input_ids"]) for i in pos]
                       for how, pos in positions.items()}
        self.probe = {(d, how): probe_cosines(encode_variant(encoder, [index.texts[i] for i in pos], **kw), index, pos)
                      for d, kw in variants.items() for how, pos in positions.items()}
        self.ndcg = {d: retrieval_metrics(encode_variant(encoder, evalset.queries, **kw), evalset.query_labels,
                                          index.embeddings, evalset.corpus_labels)["ndcg_at_10"]
                     for d, kw in variants.items()}
        # je Stichprobe: kleinste Größe, bei der alle Anfragepfade unter der Schwelle liegen
        self.needed = {how: max(int((self.probe[(d, how)] < self.threshold).argmax()) + 1 for d in variants)
                       for how in self.SAMPLES}

    def show(self, n: int) -> None:
        print(f"Stufe 2 mit {n} Texten, Schwelle {self.threshold} (der Service nimmt die ersten {self.probe_size})")
        print(f"{'':20s} {'':8s}  erste {n} (bis {max(self.tokens['erste'][:n])} Tokens)   längste {n} "
              f"(ab {min(self.tokens['längste'][:n])} Tokens)")
        for d in self.variants:
            cells = []
            for how in self.SAMPLES:
                low = self.probe[(d, how)][:n].min()
                cells.append(f"{low:6.3f} {'verweigert' if low < self.threshold else 'startet':10s}")
            print(f"{d:20s} nDCG@10 {self.ndcg[d]:.3f}   {cells[0]}           {cells[1]}")
        print("Stichprobengröße, ab der alle drei verweigert werden: " + ", ".join(
            f"{how} {self.needed[how]} {'Text' if self.needed[how] == 1 else 'Texte'}" for how in self.SAMPLES))


# --------------------------------------------------------------------------
# Blue-Green: deployment.json und ein Mini-Router
# --------------------------------------------------------------------------
def _write_json_atomic(path: Path, data: dict) -> None:
    # Erst eine Nachbardatei schreiben, dann ersetzen: Ein Leser sieht nie eine halbe Datei.
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_deployment(path: Path, active: str, urls: dict[str, str], deployments: dict) -> dict:
    """Schreibt deployment.json: aktive Farbe und je Farbe URL, Encoder und Index des Paars."""
    data = {"active": active,
            "pairs": {color: {"url": urls[color], "encoder": s.encoder_name, "index": Path(s.index_dir).name}
                      for color, s in deployments.items()}}
    _write_json_atomic(path, data)
    return data


def switch_deployment(path: Path, active: str) -> None:
    """Schaltet das aktive Paar um: eine Zeile in deployment.json, atomar ersetzt."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if active not in data["pairs"]:
        raise ValueError(f"Unbekanntes Paar '{active}', vorhanden: {list(data['pairs'])}")
    data["active"] = active
    _write_json_atomic(path, data)


class Router:
    """Mini-Router: liest bei jeder Anfrage deployment.json und leitet an das aktive Paar weiter.

    Er kennt nur Paare (eine URL je Farbe), nie Encoder oder Index einzeln. So
    gibt es keinen Zustand, in dem Anfragen des einen Encoders den Index des
    anderen treffen. Ein echter Load Balancer (Ingress, Argo Rollouts) macht
    dasselbe mit einer Service-Referenz statt einer Datei.
    """

    def __init__(self, config_path: Path, timeout: float = 30):
        import httpx

        self.config_path = Path(config_path)
        # Ruhende Verbindungen nach 1 s schließen, vor uvicorn (5 s): Sonst kann eine Anfrage auf einer Verbindung
        # landen, die der Service gerade schließt, wenn ein Client auf einem ausgelasteten Rechner länger wartet.
        self.http = httpx.Client(timeout=timeout, limits=httpx.Limits(keepalive_expiry=1.0))

    def route(self) -> tuple[str, str]:
        """Liest deployment.json und liefert (aktive Farbe, URL des Paars)."""
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        color = config["active"]
        return color, config["pairs"][color]["url"]

    def post(self, path: str, payload: dict) -> tuple[str, dict]:
        """Schickt die Anfrage an das aktive Paar. Liefert (Farbe, Antwort)."""
        color, url = self.route()
        response = self.http.post(url + path, json=payload)
        response.raise_for_status()
        return color, response.json()

    def close(self) -> None:
        self.http.close()


class BackgroundStream:
    """Anfragestrom im Hintergrund: n_clients Threads schicken Texte ohne Pause über je einen eigenen Router.

    Jede Anfrage bekommt ihre Nummer im selben Schritt, in dem ihr Router
    deployment.json liest. Die Nummern geben also die Reihenfolge der
    Routing-Entscheidungen wieder; die Anfragen selbst laufen parallel, und wer
    beim Umschalten schon unterwegs ist, wird von seiner Farbe zu Ende
    beantwortet. Fehler werden gezählt, nicht geworfen.
    """

    def __init__(self, config_path: Path, texts: list[str], n_clients: int = 2, path: str = "/similar", k: int = 1):
        import threading

        self.config_path, self.texts, self.n_clients, self.path, self.k = Path(config_path), texts, n_clients, path, k
        self.log: list[dict] = []
        self._cond = threading.Condition()
        self._next = 0
        self.done = 0
        self.in_flight = 0
        self._threads = [threading.Thread(target=self._client, daemon=True) for _ in range(n_clients)]

    def _client(self) -> None:
        router = Router(self.config_path)
        try:
            while True:
                with self._cond:
                    if self._next >= len(self.texts):
                        return
                    nr = self._next
                    self._next += 1
                    color, url = router.route()
                    self.in_flight += 1
                entry = {"nr": nr + 1, "color": color, "model": "?", "fingerprint": "?", "error": None}
                start = time.perf_counter()
                try:
                    response = router.http.post(url + self.path, json={"text": self.texts[nr], "k": self.k})
                    response.raise_for_status()
                    body = response.json()
                    entry |= {"model": body["model"], "fingerprint": body["index_fingerprint"]}
                except Exception as err:  # zählen statt abbrechen: Fehler sind ein Messergebnis
                    entry["error"] = repr(err)
                entry["ms"] = (time.perf_counter() - start) * 1000
                with self._cond:
                    self.log.append(entry)
                    self.in_flight -= 1
                    self.done += 1
                    self._cond.notify_all()
        finally:
            router.close()

    def start(self) -> "BackgroundStream":
        for thread in self._threads:
            thread.start()
        return self

    def wait_for(self, n_done: int, timeout: float = 60) -> None:
        """Wartet, bis mindestens n_done Antworten eingegangen sind."""
        with self._cond:
            if not self._cond.wait_for(lambda: self.done >= n_done, timeout):
                raise TimeoutError(f"Nach {timeout:.0f} s erst {self.done} von {n_done} Antworten.")

    def switch(self, color: str) -> dict:
        """Schaltet mitten im Strom um. Liefert, wie viele Anfragen in diesem Moment unterwegs waren.

        Das Ersetzen von deployment.json läuft unter derselben Sperre, unter der
        die Router lesen: Unter Windows schlägt os.replace fehl, solange ein
        anderer Thread die Datei geöffnet hat. Anfragen, die schon unterwegs
        sind, laufen dabei weiter; nur die nächste Routing-Entscheidung wartet
        die wenigen Mikrosekunden des Schreibens ab.
        """
        with self._cond:
            in_flight, routed = self.in_flight, self._next
            switch_deployment(self.config_path, color)
        return {"color": color, "after_routed": routed, "in_flight": in_flight}

    def join(self, timeout: float = 120) -> list[dict]:
        """Wartet auf das Ende des Stroms. Liefert das Protokoll, sortiert nach Nummer."""
        for thread in self._threads:
            thread.join(timeout)
        self.log.sort(key=lambda e: e["nr"])
        return self.log


def print_stream(log: list[dict], index_models: dict[str, str]) -> int:
    """Zeigt je Anfrage die Farbe und fasst Abschnitte gleicher Farbe zusammen.

    log: Einträge mit nr, color, model, fingerprint (aus der Antwort von /similar).
    index_models: Fingerabdruck -> Modell, mit dem der Index gebaut wurde.
    Liefert die Zahl der Antworten, in denen Encoder und Index nicht zusammenpassen
    (fehlgeschlagene Anfragen zählen nicht als Antwort).
    """
    print("Farbe je Anfrage:", "".join(e["color"][0].upper() for e in log))
    start = 0
    for i in range(1, len(log) + 1):
        if i == len(log) or log[i]["color"] != log[start]["color"]:
            e = log[start]
            print(f"  Anfrage {log[start]['nr']:2d} bis {log[i - 1]['nr']:2d}: {e['color']:5s} Encoder {e['model']:9s} "
                  f"Index {e['fingerprint']} ({index_models.get(e['fingerprint'], '?')})")
            start = i
    answered = [e for e in log if e.get("error") is None]
    mixed = sum(index_models.get(e["fingerprint"]) != e["model"] for e in answered)
    print(f"Antworten, in denen Encoder und Index nicht zusammenpassen: {mixed} von {len(answered)}")
    return mixed


# --------------------------------------------------------------------------
# Last: parallele Clients
# --------------------------------------------------------------------------
def burst(urls: list[str], n_clients: int, texts: list[str], n_per_client: int = 20, k: int = 5) -> tuple[list[float], float]:
    """Lasttest: n_clients Threads, Client i schickt an urls[i % len(urls)] n_per_client Anfragen ohne Pause.

    Jeder Thread hat einen eigenen httpx.Client (ein Client ist nicht für
    parallele Aufrufe aus mehreren Threads gedacht). Liefert (Antwortzeiten aller
    Anfragen in ms, Dauer des ganzen Tests in s); Anfragen je Sekunde sind
    len(Antwortzeiten) / Dauer.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import httpx

    if len(texts) < n_clients * n_per_client:
        raise ValueError(f"Zu wenige Texte: {len(texts)} für {n_clients} x {n_per_client} Anfragen.")
    # Alle Clients sind verbunden, bevor die Uhr läuft: Das Anlegen der Clients zählt nicht zur Dauer.
    ready = threading.Barrier(n_clients + 1, timeout=60)

    def client(i: int) -> list[float]:
        times = []
        with httpx.Client(base_url=urls[i % len(urls)], timeout=60) as http:
            ready.wait()
            for j in range(n_per_client):
                start = time.perf_counter()
                http.post("/similar", json={"text": texts[i * n_per_client + j], "k": k}).raise_for_status()
                times.append((time.perf_counter() - start) * 1000)
        return times

    with ThreadPoolExecutor(n_clients) as pool:
        futures = [pool.submit(client, i) for i in range(n_clients)]
        ready.wait()
        start = time.perf_counter()
        times = [t for future in futures for t in future.result()]
        seconds = time.perf_counter() - start
    return times, seconds


# --------------------------------------------------------------------------
# Qualität messen
# --------------------------------------------------------------------------
def engine_quality(engine, evalset: EvalSet, intents: list[str]) -> dict[str, float]:
    """nDCG@10, MRR@10, P@10, Hit@1 über alle Queries, mit Encoder und Index der Engine.

    Die Queries laufen durch engine.embed (derselbe Code wie /similar), nur im
    Batch statt als 1 000 Einzelanfragen.
    """
    label_of = {name: i for i, name in enumerate(intents)}
    corpus_labels = np.array([label_of[name] for name in engine.index.intents])
    return retrieval_metrics(engine.embed(evalset.queries), evalset.query_labels, engine.index.embeddings, corpus_labels)


def _api_scores(client, queries, query_intents, corpus_intents, k):
    n_relevant = Counter(corpus_intents)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    ndcg, hit = [], []
    for text, intent in zip(queries, query_intents):
        response = client.post("/similar", json={"text": text, "k": k})
        response.raise_for_status()
        rel = np.array([h["intent"] == intent for h in response.json()["results"]], dtype=float)
        ideal = discounts[: min(k, n_relevant[intent])].sum()
        ndcg.append(float((rel * discounts[: len(rel)]).sum() / ideal) if ideal > 0 else 0.0)
        hit.append(float(rel[0]))
    return np.array(ndcg), np.array(hit)


def api_quality(client, queries: list[str], query_intents: list[str], corpus_intents: list[str], k: int = 10) -> dict[str, float]:
    """nDCG@k und Hit@1 über die API: je Query eine Anfrage an /similar.

    client ist ein TestClient oder ein httpx.Client mit base_url. Relevant ist
    ein Treffer mit demselben Anliegen wie die Query (wie im Eval-Set).
    """
    ndcg, hit = _api_scores(client, queries, query_intents, corpus_intents, k)
    return {f"ndcg_at_{k}": float(np.mean(ndcg)), "hit_at_1": float(np.mean(hit)), "n": len(ndcg)}


def api_ndcg_per_query(client, queries: list[str], query_intents: list[str], corpus_intents: list[str],
                       k: int = 10) -> np.ndarray:
    """nDCG@k je Query über die API (für Bootstrap-Intervalle). Der Mittelwert ist api_quality."""
    return _api_scores(client, queries, query_intents, corpus_intents, k)[0]


def bootstrap_mean(values: np.ndarray, n_boot: int = 2000, seed: int = 1, level: float = 0.95) -> tuple[float, float, float]:
    """Bootstrap über die Queries: (Mittelwert, untere, obere Grenze des Intervalls)."""
    values = np.asarray(values, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(values), size=(n_boot, len(values)))
    tail = (1.0 - level) / 2 * 100
    low, high = np.percentile(values[idx].mean(axis=1), [tail, 100 - tail])
    return float(values.mean()), float(low), float(high)


def plot_pair_matrix(values: dict[tuple[str, str], float], encoders: list[str], indexes: list[str],
                     selected: tuple[str, str] | None = None, path: Path | None = None, gate: float = 0.85):
    """Matrix Encoder der Anfrage x Index mit nDCG@10; blau über dem Gate, orange darunter."""
    import matplotlib.pyplot as plt

    from .plotstyle import ACCENT, INK, MUTED, WARN

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for r, e in enumerate(encoders):
        for c, i in enumerate(indexes):
            value = values[(e, i)]
            ax.add_patch(plt.Rectangle((c - 0.48, r - 0.48), 0.96, 0.96, color=ACCENT if value >= gate else WARN,
                                       alpha=0.28, linewidth=0))
            if selected == (e, i):
                ax.add_patch(plt.Rectangle((c - 0.48, r - 0.48), 0.96, 0.96, fill=False, edgecolor=INK, linewidth=3))
            ax.text(c, r, f"{value:.3f}".replace(".", ","), ha="center", va="center", fontsize=16, color=INK,
                    fontweight="bold")
    ax.set_xlim(-0.5, len(indexes) - 0.5)
    ax.set_ylim(len(encoders) - 0.5, -0.5)
    ax.set_xticks(range(len(indexes)), [f"Index {i}" for i in indexes])
    ax.set_yticks(range(len(encoders)), [f"Anfrage {e}" for e in encoders])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("nDCG@10 je Paar aus Encoder und Index")
    ax.text(0.5, -0.16, f"blau: Gate {gate:.2f} bestanden, orange: darunter".replace(".", ","),
            transform=ax.transAxes, ha="center", color=MUTED, fontsize=10)
    if path is not None:
        fig.savefig(path)
    plt.show()


# --------------------------------------------------------------------------
# Container
# --------------------------------------------------------------------------
def container_env(settings) -> list[str]:
    """`-e`-Argumente für docker run: Pfade unter labs/ werden zu Pfaden unter /data.

    Der Container hängt labs/models nach /data/models und labs/artifacts nach
    /data/artifacts ein (nur lesend), LAB_BASE_DIR ist im Image /data.
    """
    args = []
    for key, value in service_env(settings).items():
        rel = _relative(value)
        if rel != value:
            value = "/data" if rel == "." else f"/data/{rel}"
        if (key, value) == ("LAB_BASE_DIR", "/data"):
            continue  # Standard im Image
        if key == "ENCODER_NAME" and value == Path(settings.encoder_path).name:
            continue  # Standard: Name des Encoder-Ordners
        args += ["-e", f"{key}={value}"]
    return args


def docker(*args: str, quiet: bool = False) -> str:
    """Zeigt einen docker-Befehl und führt ihn nur mit RUN_DOCKER=1 aus."""
    cmd = ["docker", *args]
    print("$", shlex.join(cmd).replace(str(LABS_DIR), "$PWD"))
    if not RUN_DOCKER:
        return ""
    if shutil.which("docker") is None:
        raise RuntimeError("docker nicht gefunden")
    done = subprocess.run(cmd, cwd=LABS_DIR, capture_output=True, text=True)
    out = (done.stdout + done.stderr).strip()
    if done.returncode != 0:
        raise RuntimeError(f"docker {args[0]} ist fehlgeschlagen:\n" + "\n".join(out.splitlines()[-15:]))
    if not quiet and out:
        print("\n".join(out.splitlines()[-6:]))
    return out


# --------------------------------------------------------------------------
# Übung 3: Rollout-Entscheidung
# --------------------------------------------------------------------------
STRATEGIES = ("Blue-Green", "Canary", "Rolling Update")


def rollout_feedback(answers: dict[str, str | None]) -> int:
    """Rückmeldung zu den eigenen Entscheidungen in Übung 3. Liefert die Zahl der Treffer."""
    from .lab3_auswertung import rollout_feedback as _feedback

    return _feedback(answers)
