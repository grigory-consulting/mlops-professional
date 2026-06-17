
import argparse
import json
import subprocess
from pathlib import Path

from prefect import flow, get_run_logger, task

HERE = Path(__file__).resolve().parent

STAGES = ["generate_data", "split", "train"]


@task(retries=2, retry_delay_seconds=10)
def dvc_repro(stage: str, force: bool = False) -> str:
    logger = get_run_logger()
    cmd = ["dvc", "repro", stage] + (["-f"] if force else [])
    logger.info("> %s", " ".join(cmd))
    result = subprocess.run(
        cmd, cwd=HERE, check=True, capture_output=True, text=True
    )
    out = (result.stdout + result.stderr).strip()
    logger.info(out or "(keine Aenderungen: Stage aus dem DVC-Cache)")
    return stage


@task
def show_metrics() -> dict:
    """Liest reports/metrics.json und loggt die Kennzahlen in den Flow-Run."""
    logger = get_run_logger()
    metrics = json.loads((HERE / "reports" / "metrics.json").read_text())
    for key, value in metrics.items():
        logger.info("%s: %s", key, value)
    return metrics


@flow(name="lab1-dvc-pipeline")
def lab1_pipeline(force: bool = False) -> dict:
    """Orchestriert die drei DVC-Stages sequenziell und gibt die Metriken zurueck.

    Direkte Task-Aufrufe in dieser Reihenfolge ergeben einen sequenziellen DAG:
    Prefect wartet auf jede Stage, bevor die naechste startet.
    """
    for stage in STAGES:
        dvc_repro(stage, force)
    return show_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="alle Stages erzwingen (dvc repro -f), statt den Cache zu nutzen",
    )
    parser.add_argument(
        "--serve", action="store_true",
        help="Flow per Zeitplan bereitstellen statt einmalig auszufuehren",
    )
    parser.add_argument(
        "--interval", type=int, default=3600,
        help="Intervall in Sekunden fuer --serve (Default: 3600 = stuendlich)",
    )
    parser.add_argument(
        "--cron", default=None,
        help="Cron-Ausdruck fuer --serve, z.B. '0 6 * * *' (taeglich 06:00). "
             "Hat Vorrang vor --interval.",
    )
    args = parser.parse_args()

    if args.serve:
        # Kleines Scheduling: stellt den Flow als Deployment bereit und triggert ihn
        # nach Zeitplan, bis Sie mit Strg+C abbrechen.
        # --cron schlaegt --interval: entweder Cron-Ausdruck oder festes Intervall.
        schedule = {"cron": args.cron} if args.cron else {"interval": args.interval}
        lab1_pipeline.serve(
            name="lab1-dvc-schedule",
            parameters={"force": False},
            **schedule,
        )
    else:
        lab1_pipeline(force=args.force)
