"""Schemas der API: Was der Service annimmt und was er zurückgibt.

Die Grenzen stehen hier an einer Stelle. Sie schützen den Service vor
Eingaben, die Speicher oder Rechenzeit sprengen (sehr lange Texte, riesige
Batches, k in der Größe des ganzen Index). Verletzt eine Anfrage eine Grenze,
antwortet FastAPI mit 422 und nennt das betroffene Feld.
"""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# Banking77: längste Anfrage 433 Zeichen. 1 000 Zeichen lassen Luft und halten
# die Tokenzahl trotzdem klein.
MAX_TEXT_CHARS = 1000
MAX_BATCH = 64
MAX_K = 50

# Leerzeichen am Rand entfernen, danach muss mindestens ein Zeichen übrig sein.
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_CHARS)]


class StrictModel(BaseModel):
    """Unbekannte Felder sind ein Fehler: Tippfehler wie `texte` fallen sofort auf."""

    model_config = ConfigDict(extra="forbid")


class EmbedRequest(StrictModel):
    texts: list[Text] = Field(
        min_length=1, max_length=MAX_BATCH, examples=[["I am still waiting on my card?"]]
    )


class EmbedResponse(BaseModel):
    model: str
    dim: int
    count: int
    embeddings: list[list[float]]


class ClassifyRequest(StrictModel):
    text: Text = Field(examples=["I am still waiting on my card?"])


class IntentScore(BaseModel):
    intent: str
    score: float


class ClassifyResponse(BaseModel):
    intent: str
    confidence: float
    top3: list[IntentScore]
    model: str
    source: str


class SimilarRequest(StrictModel):
    text: Text = Field(examples=["My card was stolen, what now?"])
    k: int = Field(default=5, ge=1, le=MAX_K)


class SimilarHit(BaseModel):
    rank: int
    text: str
    intent: str
    score: float


class SimilarResponse(BaseModel):
    model: str
    index_fingerprint: str
    results: list[SimilarHit]


class EncoderInfo(BaseModel):
    name: str
    source: str
    dim: int


class ClassifierInfo(BaseModel):
    name: str
    source: str
    n_intents: int


class IndexInfo(BaseModel):
    model_name: str
    size: int
    dim: int
    fingerprint: str
    data_hash: str


class HealthResponse(BaseModel):
    status: str
    service_version: str
    encoder: EncoderInfo
    classifier: ClassifierInfo
    index: IndexInfo
