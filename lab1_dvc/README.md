# Lab 1: DVC-Pipeline (Datenversionierung + Reproduzierbarkeit)

Eigenständige DVC-Variante von Lab 1. Während das Notebook `lab1_notebook_to_pipeline.ipynb`
das Tracking, die Registry und den Alias `champion` zeigt, demonstriert dieser Ordner
denselben Trainingslauf als **reproduzierbare DVC-Pipeline** mit drei Stages.

## Pipeline-Stages (`dvc.yaml`)

1. **generate_data**: `make_classification` erzeugt `data/raw.npz` (gesteuert über `params.yaml`)
2. **split**: stratifizierter Train/Test-Split nach `data/X_*.npy`, `data/y_*.npy`
3. **train**: RandomForest, loggt in MLflow (`./mlruns`), schreibt `models/rf.pkl` und `reports/metrics.json`

Alle Parameter stehen zentral in `params.yaml` (Seed, Datengröße, Split, RF-Hyperparameter, MLflow-Ziel).

## So führen Sie es aus

```bash
cd labs/lab1_dvc

# Abhängigkeiten (aus dem Pack-requirements): dvc, scikit-learn, mlflow, numpy, pyyaml, joblib
# in einem aktiven venv installiert haben.

# DVC einmalig initialisieren.
# Die Knowledge-Base ist kein Git-Repo, daher --no-scm:
dvc init --no-scm
# (Innerhalb eines Git-Repos stattdessen nur:  dvc init)

# Komplette Pipeline ausführen (führt nur aus, was nötig ist):
dvc repro

# Ergebnisse ansehen
dvc metrics show          # accuracy, f1, roc_auc
cat reports/metrics.json
mlflow ui --backend-store-uri ./mlruns --port 5000   # Run im Browser
```

Einzelne Stages gehen auch ohne DVC direkt:

```bash
python src/generate_data.py
python src/split.py
python src/train_rf.py
```

## Experimentieren (der eigentliche Mehrwert)

`params.yaml` ändern, z.B. `rf.n_estimators: 500`, dann erneut:

```bash
dvc repro                 # nur betroffene Stages (hier: train) laufen neu, der Rest ist gecacht
dvc params diff           # was wurde geändert
dvc metrics diff          # wie hat sich die Metrik verschoben
```

## DVC-Kommandos (Spickzettel)

### Einzelne Stages ausführen

```bash
dvc repro train                # Stage 'train' UND alle veralteten Vorstufen (split, generate_data)
dvc repro -s train             # nur 'train', Abhängigkeiten ignorieren (--single-item)
dvc repro -f train             # 'train' erzwingen, auch wenn aktuell (--force)
dvc repro --downstream split   # 'split' und alles, was davon abhängt (hier: train)
dvc repro --dry train          # zeigt nur, was liefe, ohne auszuführen
```

Die Stage-Namen stehen in `dvc.yaml`: `generate_data`, `split`, `train`.

### Status und Struktur ansehen

```bash
dvc status                     # welche Stages sind veraltet und warum
dvc dag                        # Pipeline-DAG als ASCII-Grafik
dvc dag train                  # DAG auf eine Stage fokussiert
dvc dag --dot | dot -Tpng -o dag.png   # als Bild rendern (braucht graphviz)
```

### Metriken und Parameter

```bash
dvc metrics show               # aktuelle accuracy, f1, roc_auc
dvc metrics diff               # Metrik-Änderung seit letztem Stand
dvc params diff                # geänderte Parameter aus params.yaml
```

### Outputs und Cache

```bash
dvc checkout                   # Outputs/Daten aus dem Cache in den Workspace zurückholen
dvc gc -w                      # Cache aufräumen, nur vom aktuellen Workspace Referenziertes behalten
```

Hinweis (Knowledge-Base ohne Git, `--no-scm`): `dvc metrics diff` / `dvc params diff` vergleichen gegen den zuletzt reproduzierten Stand, nicht gegen einen Git-Commit. `dvc dag` und `dvc status` funktionieren unabhängig von Git.

## Übung G: dieselbe Pipeline als Prefect-Flow orchestrieren

DVC und Prefect lösen unterschiedliche Probleme und ergänzen sich:

- **DVC** definiert die Stages (`dvc.yaml`) und sorgt für **Reproduzierbarkeit** und Caching.
- **Prefect** übernimmt die **Orchestrierung**: Reihenfolge, Retries, Logging, später Scheduling.

`flow.py` ruft pro Stage `dvc repro <stage>` auf. Prefect ersetzt DVC also nicht, sondern
steuert es. Unveränderte Stages bleiben dank DVC-Cache übersprungen, auch wenn der Flow läuft.

```bash
cd labs/lab1_dvc
dvc init --no-scm        # einmalig, falls noch nicht geschehen

python flow.py           # nur nötige Stages laufen (DVC-Cache greift)
python flow.py --force   # alle Stages erzwingen (entspricht dvc repro -f)
```

Der Flow-Run zeigt die vier Tasks (`generate_data` → `split` → `train` → `show_metrics`)
in Reihenfolge, mit Logs pro Stage und automatischem Retry, falls `train` einmal scheitert.

Optional die Prefect-UI mitlaufen lassen (zweites Terminal), um Runs im Browser zu sehen:

```bash
prefect server start                 # UI auf http://127.0.0.1:4200
# danach im ersten Terminal: python flow.py
```

Kleines Scheduling (statt einmaligem Lauf): `flow.py` kann den Flow per Zeitplan bereitstellen.

```bash
python flow.py --serve                     # stündlich ausführen (Default)
python flow.py --serve --interval 600      # alle 10 Minuten
python flow.py --serve --cron "0 6 * * *"  # täglich um 06:00 (Cron schlägt Intervall)
# läuft bis Strg+C
```

> **Wichtig:** Damit der Zeitplan tatsächlich Runs auslöst, muss ein Prefect-Server laufen
> (`prefect server start` in einem zweiten Terminal, dann `export PREFECT_API_URL=http://127.0.0.1:4200/api`).
> Ohne Server registriert `--serve` zwar das Deployment, der Scheduler feuert aber nicht
> ("Cannot schedule flows on an ephemeral server"). Der einmalige `python flow.py` braucht
> dagegen keinen Server.

**Aha-Moment:** Erst `python flow.py` (alles läuft), dann `params.yaml` ändern
(z.B. `rf.n_estimators: 500`) und `python flow.py` erneut. Prefect orchestriert weiterhin
alle vier Tasks, aber DVC führt nur `train` real aus; `generate_data` und `split` kommen
aus dem Cache. Genau das ist die Arbeitsteilung.

## Verhältnis zum Notebook

Diese Pipeline registriert **kein** Modell in der Registry. Die Registrierung von
`churn_classifier` und das Setzen des Alias `champion` (Eingabe für Lab 2 und Lab 3)
passieren im Notebook `lab1_notebook_to_pipeline.ipynb`, Übung E. Der DVC-Track hier
zeigt die orthogonale Achse: reproduzierbare, parametrisierte, cache-bare Daten- und
Trainingsschritte.
