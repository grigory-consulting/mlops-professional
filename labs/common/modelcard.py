"""Model Card aus Messwerten erzeugen und prüfen (Lab 4, Kapitel 8).

Eine Model Card (Mitchell et al. 2019, FAT* 2019, arXiv 1810.03993) beschreibt, wofür ein
Modell gedacht ist, womit es trainiert und geprüft wurde und wo seine Grenzen
liegen. Das Format folgt dem Hugging Face Hub: README.md mit YAML-Kopf und
Abschnitten im Fließtext. Die Karte entsteht aus einer Vorlage mit
Platzhaltern {{name}}; jede Kennzahl kommt aus einer Messung (aus Lab 4 oder, gekennzeichnet,
aus dem Lauf von Lab 2), keine wird abgetippt.

Zahlen im Fließtext und in Tabellen stehen mit Dezimalkomma (die Karte ist
deutsch), im YAML-Kopf maschinenlesbar mit Punkt. Die Lesefunktionen
(`card_thresholds`) verstehen beides, auch ältere Karten.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .paths import ARTIFACTS_DIR, LABS_DIR

TEMPLATE = LABS_DIR / "lab4_artifacts" / "MODEL_CARD_TEMPLATE.md"
# Karte der Vorgängerversion: beschreibt einen älteren Klassifikator mit anderen Gewichten (fester Bestand)
PREVIOUS_CARD = LABS_DIR / "lab4_artifacts" / "MODEL_CARD_vorgaenger.md"
LICENSES = {"encoder": "Apache-2.0", "data": "CC BY 4.0"}
MEASURE_KEYS = {"Zentroid-Distanz": "centroid_distance", "MMD": "mmd", "Domain-Classifier (AUC)": "domain_auc"}
TEXT_SCENARIOS = ("kein Drift", "Tippfehler", "lange Anfragen")  # Szenarien mit verändertem Text (Qualitätstabelle der Karte)


def _n(value: int) -> str:
    """Ganze Zahl mit Leerzeichen als Tausendertrenner (9 993)."""
    return f"{value:,}".replace(",", " ")


def de(value: float, fmt: str = ".3f") -> str:
    """Zahl mit Dezimalkomma für den Fließtext der Karte (0,873)."""
    return format(value, fmt).replace(".", ",")


def md_table(frame: pd.DataFrame, digits: int = 3) -> str:
    """DataFrame als Markdown-Tabelle (Index als erste Spalte), Zahlen mit Dezimalkomma."""
    def fmt(value):
        return de(value, f".{digits}f") if isinstance(value, (float, np.floating)) else str(value)

    head = "| " + " | ".join([frame.index.name or ""] + [str(c) for c in frame.columns]) + " |"
    rule = "|" + "---|" * (len(frame.columns) + 1)
    body = ["| " + " | ".join([str(i)] + [fmt(v) for v in row]) + " |"
            for i, row in zip(frame.index, frame.to_numpy(dtype=object))]
    return "\n".join([head, rule, *body])


def generate_model_card(facts: dict, output_path: Path, template_path: Path = TEMPLATE,
                        extra_sections: dict[str, str] | None = None) -> Path:
    """Füllt die Vorlage und schreibt die Karte.

    Jeder Platzhalter {{name}} braucht einen Eintrag in facts, sonst KeyError:
    Eine Karte mit Lücken soll gar nicht erst entstehen. extra_sections
    (Überschrift -> Text) werden vor dem Abschnitt Quellen eingefügt.
    """
    text = template_path.read_text(encoding="utf-8")
    for key, value in facts.items():
        text = text.replace("{{" + key + "}}", str(value))
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", text)))
    if missing:
        raise KeyError(f"Platzhalter ohne Wert: {missing}")
    if extra_sections:
        extra = "".join(f"## {title}\n\n{body.strip()}\n\n" for title, body in extra_sections.items())
        text = text.replace("## Quellen", extra + "## Quellen", 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def read_card(path: Path) -> tuple[dict, dict[str, str]]:
    """YAML-Kopf als dict und Abschnitte (Überschrift ## -> Text) einer Model Card."""
    text = Path(path).read_text(encoding="utf-8")
    _, head, body = text.split("---", 2)
    sections = {}
    for part in re.split(r"^## ", body, flags=re.M)[1:]:
        title, _, content = part.partition("\n")
        sections[title.strip()] = content.strip()
    return yaml.safe_load(head), sections


def card_accuracy(header: dict) -> float:
    """Accuracy aus dem model-index des YAML-Kopfs (so liest sie auch der Hub)."""
    return float(header["model-index"][0]["results"][0]["metrics"][0]["value"])


# ---------------------------------------------------------------- Karte prüfen (Release-Gate)

def card_hashes(text: str) -> set[str]:
    """Alle vollständigen SHA-256-Werte (64 Hexadezimalzeichen) im Text der Karte."""
    return set(re.findall(r"\b[0-9a-f]{64}\b", text))


def _number(cell: str) -> float | None:
    cell = cell.strip().replace(" ", "")
    if re.fullmatch(r"-?\d+(?:[.,]\d+)?(?:e-?\d+)?", cell):
        return float(cell.replace(",", "."))
    return None


def card_thresholds(sections: dict[str, str]) -> dict[int, dict[str, float]]:
    """Alarmschwellen aus dem Abschnitt Drift Monitoring: Fenstergröße -> {Maß -> Schwelle}.

    Versteht Spalten "Schwelle bei N Anfragen" und die ältere Form mit einer
    Spalte "Alarmschwelle", deren Fenstergröße im Text steht ("Fenster mit N Anfragen").
    Bekannte Maße heißen wie in common.monitoring (centroid_distance, mmd, domain_auc),
    andere Zeilen behalten ihren Titel.
    """
    text = sections.get("Drift Monitoring", "")
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in text.splitlines() if line.strip().startswith("|")]
    if len(rows) < 3:
        return {}
    header, body = rows[0], rows[2:]
    sizes = {}
    for col, title in enumerate(header):
        match = re.search(r"Schwelle bei (\d[\d ]*) Anfragen", title)
        if match:
            sizes[col] = int(match.group(1).replace(" ", ""))
        elif title == "Alarmschwelle":
            found = re.search(r"[Ff]enster(?:n)? mit (\d[\d ]*) Anfragen", text)
            if found:
                sizes[col] = int(found.group(1).replace(" ", ""))
    result: dict[int, dict[str, float]] = {}
    for row in body:
        key = MEASURE_KEYS.get(row[0], row[0])
        for col, size in sizes.items():
            value = _number(row[col]) if col < len(row) else None
            if value is not None:
                result.setdefault(size, {})[key] = value
    return result


def same_thresholds(card: dict[str, float], own: dict[str, float], rel: float = 0.1,
                    null_means: dict[str, float] | None = None) -> bool:
    """Stimmen die Schwellen der Karte mit den eigenen überein (je Maß auf rel genau)?

    Verglichen wird der Abstand der Schwelle von einer Basis. Mit null_means ist die Basis
    das gemessene Mittel der driftfreien Fenster (empfohlen): Dann zählt rel als Anteil
    des Abstands Schwelle zu Rauschen, und eine Schwelle nahe 0 (MMD) wird nicht zum
    Glücksspiel. Ohne null_means: bei der AUC 0,5, sonst 0.
    Fehlt ein Maß in der Karte, ist das Ergebnis False.
    """
    for key, value in own.items():
        if key not in card:
            return False
        if null_means is not None:
            base = null_means[key]
        else:
            base = 0.5 if key == "domain_auc" else 0.0
        if abs((card[key] - base) / (value - base) - 1) > rel:
            return False
    return True


def provenance_section(weights_file: str | Path, sha256: str, risk_class: str,
                       confirmed_by: str | None = None, confirmed_on: str | None = None) -> str:
    """Abschnitt Herkunft und Freigabe: Gewichtsdatei mit vollem SHA-256, Risikoklasse, Bestätigung."""
    confirmation = f"{confirmed_by}, {confirmed_on}" if confirmed_by and confirmed_on else "ausstehend"
    return "\n".join([
        f"- Gewichte: `{Path(weights_file).name}`, SHA-256 `{sha256}`",
        f"- Risikoklasse nach EU AI Act: {risk_class}",
        f"- Einstufung bestätigt (Rechtsabteilung, Datum): {confirmation}",
    ])


def risk_confirmed(sections: dict[str, str]) -> bool:
    """True, wenn die Karte eine Risikoklasse nennt und die Bestätigung Name und Datum (JJJJ-MM-TT) trägt."""
    text = sections.get("Herkunft und Freigabe", "")
    has_class = re.search(r"^- Risikoklasse nach EU AI Act: \S", text, flags=re.M) is not None
    confirmed = re.search(r"^- Einstufung bestätigt \(Rechtsabteilung, Datum\): [^,\n]+, \d{4}-\d{2}-\d{2}\s*$", text, flags=re.M)
    return has_class and confirmed is not None


# ---------------------------------------------------------------- Karte erzeugen

def _baseline_accuracy() -> float | None:
    """Test-Accuracy der Baseline aus Lab 2 (logistische Regression auf Embeddings), falls ein Lauf vorliegt."""
    for folder in ("lab2",):
        path = ARTIFACTS_DIR / folder / "results.json"
        try:
            return float(json.loads(path.read_text(encoding="utf-8"))["baseline_minilm_logreg"]["test_accuracy"])
        except (OSError, KeyError, ValueError, TypeError):
            continue
    return None


def data_stats(train: pd.DataFrame, test: pd.DataFrame, share: float = 0.95) -> dict[str, int]:
    """Kennzahlen der Daten für die Karte: Wortzahl der Trainingsanfragen und Beispiele je Anliegen.

    words_share: kleinste Wortzahl, die mindestens share der Trainingsanfragen nicht überschreiten.
    """
    words = train["text"].str.split().str.len().to_numpy()
    train_counts, test_counts = train["label"].value_counts(), test["label"].value_counts()
    return {"words_median": int(np.median(words)), "words_share": int(np.sort(words)[int(np.ceil(share * len(words))) - 1]),
            "share_pct": int(round(share * 100)), "words_max": int(words.max()),
            "train_min": int(train_counts.min()), "train_max": int(train_counts.max()),
            "test_min": int(test_counts.min()), "test_max": int(test_counts.max())}


def base_card_facts(*, model_name: str, classifier_info: dict, encoder_name: str, encoder_hub_id: str,
                    accuracy: float, hit_at_1: float, n_eval: int, n_train: int, n_test: int, data_hash: str,
                    scenario_quality: pd.DataFrame, drift_findings: str, stats: dict[str, int],
                    caveats_extra: list[str] | None = None,
                    quantitative_analyses: str = "Qualität je Gruppe: noch nicht ausgewertet.") -> dict:
    """Alle Einträge der Vorlage. Feste Texte (Zweck, Grenzen, Lizenzen) plus die übergebenen Messwerte.

    Vergleichssätze entstehen aus den Werten (nicht fest eingetragen); die Baseline stammt aus
    dem Lauf von Lab 2 (results.json) und ist in der Karte so gekennzeichnet.
    """
    search_text = ("Die Suche (Hit@1) liegt über dem Klassifikator. " if hit_at_1 > accuracy else
                   "Der Klassifikator liegt über der Suche (Hit@1). ")
    baseline = _baseline_accuracy()
    if baseline is None:
        baseline_text = "Ein Vergleich mit der Baseline aus Lab 2 liegt nicht vor. "
    else:
        side = "über" if baseline > accuracy else "unter"
        baseline_text = (f"Die Baseline aus Lab 2 (logistische Regression auf den Embeddings, Test-Accuracy {de(baseline)}, "
                         f"gemessen in Lab 2, nicht in Lab 4) liegt {side} dem Klassifikator. ")
    return {
        "model_name": model_name,
        "model_kind": classifier_info["kind"],
        "library_name": classifier_info["library_name"],
        "base_model": classifier_info["base_model"],
        "base_model_license": classifier_info["base_model_license"],
        "encoder_name": encoder_name,
        "encoder_hub_id": encoder_hub_id,
        "encoder_license": LICENSES["encoder"],
        "accuracy": round(accuracy, 4),
        "card_date": date.today().isoformat(),
        "owner": "Team Kundenservice-KI (im Kurs: Ihr Name)",
        "data_hash": data_hash,
        "summary": (f"`{model_name}` ordnet kurze englische Kundenanfragen einer Bank einem von 77 Anliegen zu. "
                    f"Quelle des Modells: {classifier_info['source']}."),
        "intended_use": "\n".join([
            "- Vorschlag für das Routing einer Kundenanfrage an das zuständige Team, zusammen mit der Suche nach ähnlichen früheren Anfragen.",
            "- Nutzer: Mitarbeitende im Kundenservice. Die Entscheidung über den Vorgang bleibt beim Menschen.",
            f"- Eingabe: eine kurze englische Anfrage (Trainingsanfragen: Median {stats['words_median']} Wörter, "
            f"{stats['share_pct']} % höchstens {stats['words_share']} Wörter, längste {stats['words_max']}).",
        ]),
        "out_of_scope": "\n".join([
            "- Andere Sprachen als Englisch.",
            "- Anliegen außerhalb der 77 festen Klassen und andere Branchen.",
            "- Lange Texte mit mehreren Anliegen (E-Mails, Briefe).",
            "- Automatische Entscheidungen mit Wirkung für Kunden (z. B. Kartensperre, Kredit) ohne menschliche Prüfung.",
        ]),
        "training_data": "\n".join([
            "- Banking77 (Casanueva et al. 2020, PolyAI), Lizenz CC BY 4.0: Namensnennung ist Pflicht.",
            f"- Im Kurs: {_n(n_train)} Trainings- und {_n(n_test)} Testanfragen, 77 Anliegen, Englisch.",
            f"- Lizenzkette: Basismodell {classifier_info['base_model_license']}, Embedding-Modell {LICENSES['encoder']}, "
            "Daten CC BY 4.0. Alle erlauben die kommerzielle Nutzung.",
            "- Weitergabe des Modells (auch an eine Tochterfirma): Lizenztext Apache 2.0 beilegen, NOTICE-Hinweise des "
            "Basismodells erhalten (falls vorhanden), geänderte Dateien kennzeichnen (Apache 2.0, Abschnitt 4); "
            "Banking77 mit Quelle, Lizenz und Hinweis auf Änderungen nennen (CC BY 4.0). Keine Rechtsberatung.",
        ]),
        "evaluation": "\n".join([
            f"Gemessen in Lab 4 auf {_n(n_eval)} Testanfragen (vom Modell nicht gesehen):",
            "",
            "| Kennzahl | Wert |",
            "|---|---|",
            f"| Accuracy des Klassifikators | {de(accuracy)} |",
            f"| Hit@1 der Suche (`{encoder_name}`, Index: alle Trainingsanfragen) | {de(hit_at_1)} |",
            "",
            search_text + baseline_text
            + "Diese Karte beschreibt den Klassifikator, den der Service lädt; ob er champion bleibt, entscheidet das Gate.",
            "",
            "Qualität bei verändertem Text (Fenster mit 500 Anfragen):",
            "",
            md_table(scenario_quality.rename(index={"lange Anfragen": "lange Anfragen (zwei Anfragen zum selben Anliegen)"})
                     .rename_axis("Szenario")),
            "",
            "Die langen Anfragen hängen zwei Anfragen zum selben Anliegen aneinander. Über lange Texte mit mehreren "
            "Anliegen sagen sie nichts (siehe Out-of-Scope). Ein veränderter Themenmix ändert nicht den Text, sondern "
            "die Anteile der Anliegen; seine Wirkung folgt aus der Qualität je Anliegen-Gruppe (Quantitative Analyses).",
        ]),
        "metrics": "\n".join([
            f"- **Accuracy:** Anteil richtig zugeordneter Anfragen. Geeignet, weil das Testset ausgewogen ist "
            f"({stats['test_min']} bis {stats['test_max']} Anfragen je Anliegen).",
            "- **Hit@1:** Anteil der Anfragen, deren ähnlichste frühere Anfrage dasselbe Anliegen hat.",
        ]),
        "quantitative_analyses": quantitative_analyses,
        "drift_findings": drift_findings,
        "ethical_considerations": "\n".join([
            "- Im Betrieb enthalten Kundenanfragen personenbezogene Daten. Das Loggen für die Drift-Überwachung braucht eine Rechtsgrundlage, eine Löschfrist und Zugriffsschutz.",
            "- Falsch zugeordnete Anfragen verlängern die Bearbeitung. Betroffen sind vor allem Kunden, die anders schreiben als die Trainingsdaten (Tippfehler, andere Erstsprache).",
            "- Das Modell trifft keine Entscheidung über Kunden. Es darf nicht ohne neue Prüfung dafür eingesetzt werden.",
        ]),
        "caveats": "\n".join([
            "- Das Modell kennt genau 77 Anliegen und antwortet immer mit einem davon. Neue Anliegen erkennt es nicht.",
            f"- Die Trainingsdaten sind unausgewogen ({stats['train_min']} bis {stats['train_max']} Beispiele je Anliegen).",
            *(caveats_extra or []),
            "- Bei einem Wechsel des Embedding-Modells müssen Suchindex, Referenzfenster und Schwellen neu berechnet werden.",
            "- Die Kennzahlen stammen aus einem öffentlichen Datensatz. Vor dem Einsatz mit eigenen Anfragen neu messen.",
        ]),
    }


def drift_section(thresholds: dict[int, dict[str, float]], ref_size: int, encoder: str,
                  notes: list[str] | None = None) -> str:
    """Abschnitt Drift Monitoring: Schwellen je Fenstergröße (Spalten) und Befunde als Liste.

    thresholds: Fenstergröße -> {Maß -> Schwelle}, z. B. {500: {...}, 100: {...}}.
    """
    titles = {key: title for title, key in MEASURE_KEYS.items()}
    sizes = list(thresholds)
    keys = list(thresholds[sizes[0]])
    lines = [
        f"Überwacht werden Fenster mit aufeinanderfolgenden Anfragen gegen ein Referenzfenster mit {_n(ref_size)} Anfragen "
        f"(13 je Anliegen, nicht aus dem Training). Embeddings: {encoder}.",
        "",
        "| Maß | " + " | ".join(f"Schwelle bei {size} Anfragen" for size in sizes) + " |",
        "|---|" + "---|" * len(sizes),
        *[f"| {titles.get(key, key)} | " + " | ".join(de(thresholds[size][key], ".4g") for size in sizes) + " |" for key in keys],
        "",
        "Die Schwellen sind Mittelwert plus drei Standardabweichungen aus 200 driftfreien Fenstern "
        "und gelten nur für die genannte Fenstergröße.",
    ]
    if notes:
        lines += ["", *[f"- {note}" for note in notes]]
    lines += ["", "Ein Drift-Alarm löst eine Prüfung aus (Stichprobe labeln, Qualität messen), kein automatisches Retraining."]
    return "\n".join(lines)


def lab4_card_facts(*, classifier_info: dict, train: pd.DataFrame, test: pd.DataFrame, ref_sig: pd.DataFrame,
                    log, windows: dict, embedding: pd.DataFrame, thresholds: dict,
                    window_size: int = 500) -> dict:
    """Einträge der Model Card aus den Messwerten von Lab 4 (Demo und Übungen nutzen dieselbe Fassung).

    Qualität auf allen unveränderten Testanfragen (Referenz und Pool), je Szenario mit
    verändertem Text (TEXT_SCENARIOS) auf dessen Fenster; unter Drift Monitoring die
    Embedding-Maße aller Szenarien. thresholds: entweder {Maß -> Schwelle} für window_size Anfragen
    oder {Fenstergröße -> {Maß -> Schwelle}} für mehrere Fenstergrößen.
    """
    from .data import dataset_hash
    from .models import EMBEDDING_MODELS
    from .monitoring import MEASURE_TITLES
    from .tracking import CLASSIFIER_NAME, ENCODER_NAME

    by_size = thresholds if all(isinstance(key, int) for key in thresholds) else {window_size: thresholds}
    keys = list(next(iter(by_size.values())))
    clean = pd.concat([ref_sig, log.signals.iloc[log.pool]], ignore_index=True)
    quality = {"accuracy": "correct", "hit_at_1": "hit1", "mean_confidence": "confidence"}
    # Qualitätstabelle: Szenarien, die den Text selbst verändern (die Themenverschiebung ändert nur die Anteile)
    scenario_quality = pd.DataFrame({name: {key: float(log.signals.iloc[idx][col].mean()) for key, col in quality.items()}
                                     for name, idx in windows.items() if name in TEXT_SCENARIOS}).T
    low = clean["confidence"] < 0.5
    notes = [f"{name}: " + ", ".join(f"{MEASURE_TITLES[key]} {de(embedding.loc[name, key], '.3g')}" for key in keys)
             for name in windows]
    encoder = f"`{ENCODER_NAME}` (`minilm-l6`)"
    return base_card_facts(
        model_name=CLASSIFIER_NAME, classifier_info=classifier_info, encoder_name=ENCODER_NAME,
        encoder_hub_id=EMBEDDING_MODELS["minilm-l6"], accuracy=float(clean["correct"].mean()),
        hit_at_1=float(clean["hit1"].mean()), n_eval=len(clean), n_train=len(train), n_test=len(test),
        data_hash=dataset_hash(train, test)[:16], scenario_quality=scenario_quality,
        drift_findings=drift_section(by_size, len(ref_sig), encoder, notes), stats=data_stats(train, test),
        caveats_extra=[
            f"- Vertippte Anfragen senken die Accuracy deutlich (gemessen: {de(scenario_quality.loc['Tippfehler', 'accuracy'])} "
            f"statt {de(clean['correct'].mean())}).",
            f"- Anfragen mit Konfidenz unter 0,5 ({low.mean() * 100:.0f} % der Testanfragen) sind nur zu {clean.loc[low, 'correct'].mean() * 100:.0f} % "
            f"richtig zugeordnet, die übrigen zu {clean.loc[~low, 'correct'].mean() * 100:.0f} %. Empfehlung: unter 0,5 an einen Menschen weiterleiten.",
        ],
    )
