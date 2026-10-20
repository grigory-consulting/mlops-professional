# MLOps: Von der Entwicklung zum produktiven Einsatz

Folien und Lab-Material zur dreitägigen Schulung (gfu).

**Folien:** https://grigory-consulting.github.io/mlops-professional/
(Pfeiltasten zum Blättern, `Esc` für die Übersicht, `F` für Vollbild)

## Lab 1 vorbereiten

Python 3.11, nur CPU. Einmalig mit Netzzugang:

```bash
git clone https://github.com/grigory-consulting/mlops-professional.git
cd mlops-professional
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd labs
python prefetch.py                 # Banking77 und Modelle, rund 0,6 GB
python prefetch.py --check
```

MLflow-Oberfläche (im Ordner `labs/`):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db    # Port 5000 belegt (macOS AirPlay): --port 5001
```

Lab 1 braucht kein Docker.

## Airflow in Docker (Lab 2, auch unter Windows)

Airflow läuft unter Windows nicht nativ, deshalb im Container. Voraussetzungen:

- Docker Desktop läuft, `prefetch.py` ist einmal durchgelaufen (`labs/data/`, `labs/models/` gefüllt).
- Docker Desktop, *Settings > Resources > Advanced*: mindestens 4 GB Arbeitsspeicher.
- Nur Windows ohne WSL (Hyper-V-Backend): *Settings > Resources > File sharing* den Kursordner
  (z. B. `C:\mlops-professional`) hinzufügen und *Apply & restart*. Sonst ist `labs/` im Container leer.

Starten (PowerShell, Terminal oder Git Bash, im Ordner `labs/airflow_docker/`):

```bash
docker compose up --build          # erster Build rund 5 min, danach Sekunden
```

Oberfläche: http://localhost:8080 (ohne Login), DAG `banking_intent_pipeline` aktivieren und starten.
Ohne Oberfläche einmal durchlaufen lassen:

```bash
docker compose run --rm airflow dags test banking_intent_pipeline
```

Beenden: `Strg+C`, danach `docker compose down`. Ergebnisse landen in `labs/model/` und `labs/reports/`,
MLflow-Läufe aus dem Container in `labs/mlflow-docker.db`
(Oberfläche: `mlflow ui --backend-store-uri sqlite:///mlflow-docker.db`).

## Wenn etwas nicht läuft

| Meldung oder Beobachtung | Abhilfe |
|---|---|
| `No module named 'common'` | Notebooks im Ordner `labs/` ablegen und von dort öffnen |
| Download bricht ab oder hängt | `huggingface.co` im Netz gesperrt: Archiv von `labs/data/` und `labs/models/` vom Trainer holen, entpacken, `python prefetch.py --check` |
| `prefetch.py --check` meldet fehlende Modelle | `python prefetch.py` erneut starten |
| Python 3.12 oder neuer | meist unkritisch; bei Installationsfehlern eine 3.11-Umgebung anlegen |
| Windows: Pfad zu lang | Kursmaterial nah an `C:\` ablegen |
| Airflow zeigt keinen DAG, `labs/` im Container leer | Docker Desktop: Kursordner unter *Settings > Resources > File sharing* freigeben |
| Airflow: `train` mit `return code -9`, dann `up_for_retry` | Zu wenig Speicher: Docker Desktop auf mindestens 4 GB stellen |
| `port is already allocated` (8080) | PowerShell: `$env:AIRFLOW_PORT="8081"`, dann `docker compose up`; Oberfläche auf Port 8081 |
