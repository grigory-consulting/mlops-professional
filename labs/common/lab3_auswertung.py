"""Auswertung zu Lab 3, Übung 3: Rückmeldung zur Rollout-Entscheidung und Vergleichsmessung.

Dieses Modul enthält die Musterlösung von Übung 3. Öffnen Sie es erst, nachdem
Sie Ihre eigene Entscheidung eingetragen und die Rückmeldung gesehen haben.
"""
from __future__ import annotations

from .metrics import retrieval_metrics
from .models import encode

# "muster": Antworten, die als Treffer zählen (mehrere, wo die Folie beide nennt);
# "vertretbar": Antworten mit Einschränkung, je mit Begründung.
ROLLOUT_SOLUTION = {
    "minilm_feinjustiert": {
        "muster": ("Blue-Green",),
        "vertretbar": {
            "Canary": "Nur als vollständiges Paar aus neuem Encoder und eigenem, neu gebautem Index; das braucht "
                      "dieselbe doppelte Infrastruktur wie Blue-Green.",
            "Rolling Update": "Nur, wenn jede Instanz ihr eigenes Paar mitbringt (Modell und Index im Image, "
                              "Dockerfile.baked): Dann ersetzt das Update Paar für Paar, ohne Mischantworten. Der "
                              "Rückweg ist langsamer als ein Umschalten, und während des Rollouts sehen Nutzer zwei "
                              "verschiedene Ergebnislisten. Mit einem gemeinsamen Index wie auf der Folie geht es "
                              "nicht.",
        },
        "warum": "Neue Gewichte heißen neue Vektoren, auch wenn der Name minilm-l6 bleibt: Der alte Index gehört nicht "
                 "mehr zum Encoder, wie beim getarnten Encoder aus Demo 4. Stufe 1 der Startprüfung lässt ihn durch; "
                 "Stufe 2 verweigert neuen Instanzen mit dem alten Index den Start, sobald die Stichprobe unter 0,99 "
                 "fällt. Mit einem gemeinsamen Index bliebe ein Rolling Update also bei der ersten Instanz stehen. "
                 "Deshalb neuer Index neben dem alten und das Paar in einem Schritt umschalten.",
    },
    "klassifikator_max_length": {
        "muster": ("Canary", "Rolling Update"),
        "vertretbar": {"Blue-Green": "Funktioniert, kostet aber doppelte Infrastruktur, obwohl kein Zustand "
                                     "umgeschaltet werden muss."},
        "warum": "Die Einstellung ändert nur den Anfragepfad des Klassifikators, ein Index hängt nicht daran: Jede "
                 "Antwort hängt nur von der Anfrage ab, zwei Versionen dürfen nebeneinander antworten. Anders als "
                 "beim Encoder gibt es keinen gespeicherten Zustand, der zur alten Einstellung passen muss. Canary "
                 "zeigt zusätzlich an einem kleinen Anteil, ob lange Anfragen schlechter zugeordnet werden.",
    },
    "index_ergaenzt": {
        "muster": ("Rolling Update",),
        "vertretbar": {"Canary": "Möglich, aber im Mischbetrieb fehlen nur einigen Antworten die neuesten Einträge; "
                                 "zu beobachten gibt es wenig.",
                       "Blue-Green": "Funktioniert, kostet aber doppelte Infrastruktur ohne Nutzen: Alter und neuer "
                                     "Index liegen im selben Vektorraum."},
        "warum": "Gleicher Encoder, also derselbe Vektorraum: Jede Instanz mit altem oder neuem Index antwortet "
                 "richtig, die alten kennen nur die neuen Einträge noch nicht. Die Startprüfung jeder neuen "
                 "Instanz läuft gegen den neuen Index.",
    },
    "bugfix_max_seq_length": {
        "muster": ("Rolling Update",),
        "vertretbar": {"Canary": "Möglich, aber der Fix stellt nur den Zustand her, zu dem der Index passt; "
                                 "jede neue Instanz antwortet besser als eine alte."},
        "warum": "Der Index bleibt unverändert, der Anfragepfad passt nach dem Fix wieder zu ihm. Alte und neue "
                 "Instanzen suchen im selben Vektorraum, ein Mischbetrieb schadet nicht.",
    },
}


def rollout_feedback(answers: dict[str, str | None]) -> int:
    """Vergleicht die Rollout-Entscheidungen mit der Musterlösung. Liefert die Zahl der Treffer."""
    hits = 0
    for key, solution in ROLLOUT_SOLUTION.items():
        answer = answers.get(key)
        muster = " oder ".join(solution["muster"])
        if answer in solution["muster"]:
            hits += 1
            verdict = "Treffer."
        elif answer in solution["vertretbar"]:
            verdict = f"vertretbar (Muster: {muster}). {solution['vertretbar'][answer]}"
        elif answer is None:
            verdict = f"keine Antwort (Muster: {muster})."
        else:
            verdict = f"abweichend (Muster: {muster})."
        print(f"{key}: {answer} -> {verdict}\n    {solution['warum']}")
    return hits


def query_path_table(encoders: dict, evalset, index, blue: str = "minilm-l6", green: str = "bge-small") -> dict[str, float]:
    """nDCG@10 im Index von blue für drei Anfragepfade, alle Queries des Eval-Sets.

    Zeigt, was Nutzer bekommen, solange eine Instanz mit diesem Anfragepfad im
    alten Index sucht: Anfragepfad mit 24 Tokens (vor dem Bugfix), Anfragepfad
    nach dem Fix, und als äußerster Fall eines Encoders mit neuen Gewichten unter
    gleichem Namen der getarnte Encoder aus Demo 4 (bge-small als minilm-l6).
    """
    from .encoder_eval import encode_variant

    paths = {
        "Anfragepfad mit 24 Tokens (vor dem Fix)": encode_variant(encoders[blue], evalset.queries, max_seq_length=24),
        "Anfragepfad nach dem Fix": encode(encoders[blue], evalset.queries),
        "ganz andere Gewichte, gleicher Name (getarnt, Demo 4)": encode(encoders[green], evalset.queries),
    }
    result = {}
    for name, q in paths.items():
        result[name] = retrieval_metrics(q, evalset.query_labels, index.embeddings, evalset.corpus_labels)["ndcg_at_10"]
        print(f"{name:54s} nDCG@10 {result[name]:.3f}")
    return result
