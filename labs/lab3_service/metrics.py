"""Zähler und Latenzen im Prometheus-Textformat, ohne zusätzliche Bibliothek.

Für den Kurs reicht das: Man sieht, woraus eine Metrik besteht (Name, Labels,
Wert) und wie ein Histogramm Latenzen in Klassen zählt. Im Betrieb nimmt man
`prometheus_client`, das dasselbe Format erzeugt.
"""
from __future__ import annotations

import threading
from collections import defaultdict

BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
PREFIX = "banking_service"


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, int], int] = defaultdict(int)
        self._buckets: dict[str, list[int]] = defaultdict(lambda: [0] * len(BUCKETS))
        self._sum: dict[str, float] = defaultdict(float)
        self._count: dict[str, int] = defaultdict(int)
        self._texts = 0

    def observe(self, endpoint: str, status: int, seconds: float) -> None:
        with self._lock:
            self._requests[(endpoint, status)] += 1
            self._sum[endpoint] += seconds
            self._count[endpoint] += 1
            for i, upper in enumerate(BUCKETS):
                if seconds <= upper:
                    self._buckets[endpoint][i] += 1

    def add_texts(self, n: int) -> None:
        with self._lock:
            self._texts += n

    def render(self, info: dict[str, str]) -> str:
        """Erzeugt den Text, den Prometheus von /metrics abholt."""
        labels = ",".join(f'{key}="{value}"' for key, value in info.items())
        out = [
            f"# HELP {PREFIX}_info Geladene Modelle und Index (Wert ist immer 1).",
            f"# TYPE {PREFIX}_info gauge",
            f"{PREFIX}_info{{{labels}}} 1",
            f"# HELP {PREFIX}_requests_total Anfragen je Endpunkt und Statuscode.",
            f"# TYPE {PREFIX}_requests_total counter",
        ]
        with self._lock:
            for (endpoint, status), n in sorted(self._requests.items()):
                out.append(f'{PREFIX}_requests_total{{endpoint="{endpoint}",status="{status}"}} {n}')
            out += [
                f"# HELP {PREFIX}_texts_embedded_total Texte, die der Encoder verarbeitet hat.",
                f"# TYPE {PREFIX}_texts_embedded_total counter",
                f"{PREFIX}_texts_embedded_total {self._texts}",
                f"# HELP {PREFIX}_request_duration_seconds Dauer der Anfragen in Sekunden.",
                f"# TYPE {PREFIX}_request_duration_seconds histogram",
            ]
            for endpoint in sorted(self._count):
                name = f"{PREFIX}_request_duration_seconds"
                for upper, n in zip(BUCKETS, self._buckets[endpoint]):
                    out.append(f'{name}_bucket{{endpoint="{endpoint}",le="{upper}"}} {n}')
                out.append(f'{name}_bucket{{endpoint="{endpoint}",le="+Inf"}} {self._count[endpoint]}')
                out.append(f'{name}_sum{{endpoint="{endpoint}"}} {self._sum[endpoint]:.6f}')
                out.append(f'{name}_count{{endpoint="{endpoint}"}} {self._count[endpoint]}')
        return "\n".join(out) + "\n"
