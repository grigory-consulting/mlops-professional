"""Kleiner Abnahmetest gegen einen laufenden Service (nur Standardbibliothek).

Aufruf:
    python lab3_service/smoke_test.py http://127.0.0.1:8000

Prüft je Endpunkt einen Gutfall und Fehlerfälle (422). Exit-Code 0, wenn alles
stimmt, sonst 1. Geeignet als letzter Schritt nach `docker run` oder in CI.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request


def call(base_url: str, path: str, payload: dict | None = None) -> tuple[int, object]:
    """Schickt GET (ohne payload) oder POST (mit payload). Liefert Statuscode und Antwort."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(base_url + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status, body = response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        status, body = err.code, err.read().decode("utf-8")
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body


def main(base_url: str) -> int:
    checks: list[tuple[str, bool]] = []

    status, health = call(base_url, "/health")
    checks.append(("GET /health liefert 200 und status ok", status == 200 and health["status"] == "ok"))
    checks.append(("Encoder und Index tragen denselben Modellnamen",
                   status == 200 and health["encoder"]["name"] == health["index"]["model_name"]))

    status, body = call(base_url, "/embed", {"texts": ["I am still waiting on my card?", "What is the exchange rate?"]})
    checks.append(("POST /embed liefert 2 Vektoren passender Dimension",
                   status == 200 and body["count"] == 2 and len(body["embeddings"][0]) == body["dim"]))

    status, body = call(base_url, "/classify", {"text": "I am still waiting on my card?"})
    checks.append(("POST /classify liefert Anliegen, Konfidenz und Top-3",
                   status == 200 and len(body["top3"]) == 3 and 0.0 <= body["confidence"] <= 1.0))

    status, body = call(base_url, "/similar", {"text": "My card was stolen, what now?", "k": 3})
    scores = [hit["score"] for hit in body["results"]] if status == 200 else []
    checks.append(("POST /similar liefert 3 Treffer, absteigend sortiert",
                   len(scores) == 3 and scores == sorted(scores, reverse=True)))

    for name, path, payload in [
        ("leere Textliste", "/embed", {"texts": []}),
        ("Text nur aus Leerzeichen", "/classify", {"text": "   "}),
        ("k zu groß", "/similar", {"text": "card", "k": 10_000}),
    ]:
        status, _ = call(base_url, path, payload)
        checks.append((f"POST {path} lehnt ab ({name}): 422", status == 422))

    status, body = call(base_url, "/metrics")
    checks.append(("GET /metrics zählt Anfragen", status == 200 and "banking_service_requests_total" in str(body)))

    for name, ok in checks:
        print(("ok      " if ok else "FEHLER  ") + name)
    failed = sum(not ok for _, ok in checks)
    print("alle Prüfungen bestanden" if failed == 0 else f"{failed} Prüfung(en) fehlgeschlagen")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000"))
