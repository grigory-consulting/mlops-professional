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

## Wenn etwas nicht läuft

| Meldung oder Beobachtung | Abhilfe |
|---|---|
| `No module named 'common'` | Notebooks im Ordner `labs/` ablegen und von dort öffnen |
| Download bricht ab oder hängt | `huggingface.co` im Netz gesperrt: Archiv von `labs/data/` und `labs/models/` vom Trainer holen, entpacken, `python prefetch.py --check` |
| `prefetch.py --check` meldet fehlende Modelle | `python prefetch.py` erneut starten |
| Python 3.12 oder neuer | meist unkritisch; bei Installationsfehlern eine 3.11-Umgebung anlegen |
| Windows: Pfad zu lang | Kursmaterial nah an `C:\` ablegen |
