"""FastAPI-Service für den Fall: Encoder, Klassifikator und Vektorindex hinter einer API.

Start (im Ordner labs/):
    uvicorn lab3_service.app:app --host 127.0.0.1 --port 8000

Konfiguration über Umgebungsvariablen, siehe `engine.Settings`. Ohne Variablen
lädt der Service `models/minilm-l6` und `artifacts/lab3/index_minilm-l6`.
Passen Encoder und Index nicht zusammen, startet der Service nicht.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

from .engine import Engine, Settings, load_engine
from .metrics import Metrics
from .schemas import (
    ClassifyRequest,
    ClassifyResponse,
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    SimilarRequest,
    SimilarResponse,
)

SERVICE_VERSION = "1.0.0"
# Nur diese Pfade werden als Label gezählt. Sonst erzeugt jeder Tippfehler in
# einer URL eine neue Zeitreihe in Prometheus.
ENDPOINTS = ("/health", "/embed", "/classify", "/similar", "/metrics")
log = logging.getLogger("uvicorn.error")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Baut die App. Tests übergeben eigene Settings, uvicorn nutzt die Umgebungsvariablen."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Einmal beim Start laden, nicht je Anfrage. Scheitert das Laden oder
        # die Prüfung von Encoder und Index, nimmt der Service keine Anfragen an.
        start = time.perf_counter()
        engine = load_engine(settings or Settings.from_env())
        app.state.engine = engine
        log.info(
            "Bereit nach %.1f s: Encoder %s, Index %s (%d Einträge), Klassifikator %s",
            time.perf_counter() - start,
            engine.settings.encoder_name,
            engine.index.fingerprint,
            len(engine.index.texts),
            engine.classifier.source,
        )
        yield
        app.state.engine = None

    app = FastAPI(
        title="Banking-Service: Encoder und Klassifikator",
        version=SERVICE_VERSION,
        description=(
            "Kundenanfragen einer Bank: Embeddings berechnen, Anliegen bestimmen "
            "und ähnliche frühere Anfragen finden (Datensatz Banking77)."
        ),
        lifespan=lifespan,
    )
    metrics = Metrics()

    @app.middleware("http")
    async def measure(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        path = request.url.path if request.url.path in ENDPOINTS else "other"
        metrics.observe(path, response.status_code, time.perf_counter() - start)
        return response

    def engine_of(request: Request) -> Engine:
        return request.app.state.engine

    @app.get("/health", response_model=HealthResponse, tags=["Betrieb"], summary="Zustand und geladene Versionen")
    def health(request: Request):
        return {"status": "ok", "service_version": SERVICE_VERSION, **engine_of(request).health()}

    @app.get("/metrics", response_class=PlainTextResponse, tags=["Betrieb"], summary="Zähler und Latenzen für Prometheus")
    def metrics_endpoint(request: Request):
        engine = engine_of(request)
        return metrics.render({
            "version": SERVICE_VERSION,
            "encoder": engine.settings.encoder_name,
            "classifier_source": engine.classifier.source,
            "index_fingerprint": engine.index.fingerprint,
        })

    @app.post("/embed", response_model=EmbedResponse, tags=["Modelle"], summary="Texte in Vektoren übersetzen")
    def embed(body: EmbedRequest, request: Request):
        engine = engine_of(request)
        vectors = engine.embed(body.texts)
        metrics.add_texts(len(body.texts))
        return {
            "model": engine.settings.encoder_name,
            "dim": engine.dim,
            "count": len(body.texts),
            "embeddings": vectors.tolist(),
        }

    @app.post("/classify", response_model=ClassifyResponse, tags=["Modelle"], summary="Anliegen einer Anfrage bestimmen")
    def classify(body: ClassifyRequest, request: Request):
        return engine_of(request).classify(body.text)

    @app.post("/similar", response_model=SimilarResponse, tags=["Modelle"], summary="Ähnliche frühere Anfragen finden")
    def similar(body: SimilarRequest, request: Request):
        engine = engine_of(request)
        metrics.add_texts(1)
        return {
            "model": engine.settings.encoder_name,
            "index_fingerprint": engine.index.fingerprint,
            "results": engine.similar(body.text, body.k),
        }

    return app


app = create_app()
