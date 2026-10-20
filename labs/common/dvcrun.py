"""DVC-Pipeline von Lab 2 in einer Arbeitskopie ausführen (ohne Git, ohne Netz).

Das Original unter labs/lab2_pipeline/ bleibt unberührt. Eine Arbeitskopie
enthält die Stufen-Skripte, dvc.yaml und params.yaml; `dvc init --no-scm`
macht daraus ein DVC-Projekt ohne Git-Repository.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from .paths import LABS_DIR

PIPELINE_SRC = LABS_DIR / "lab2_pipeline"
STAGES = ("validate_data", "train", "evaluate", "gate", "register")


def make_workcopy(dest: str | Path, **overrides) -> Path:
    """Legt eine frische Arbeitskopie der Pipeline an und setzt Parameter.

    overrides mit Punktnotation als Schlüssel ohne Punkte: train__epochs=1
    setzt train.epochs. Ein vorhandener Ordner wird ersetzt.
    """
    dest = Path(dest)
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    for path in list(PIPELINE_SRC.glob("*.py")) + [PIPELINE_SRC / "dvc.yaml", PIPELINE_SRC / "params.yaml"]:
        if path.name != "reference_run.py":
            shutil.copy(path, dest / path.name)
    for key, value in overrides.items():
        set_param(dest, key.replace("__", "."), value)
    dvc(dest, "init", "--no-scm", "-q")
    dvc(dest, "config", "core.analytics", "false")
    dvc(dest, "config", "core.check_update", "false")
    return dest


def get_param(workdir: str | Path, key: str):
    params = yaml.safe_load((Path(workdir) / "params.yaml").read_text(encoding="utf-8"))
    for part in key.split("."):
        params = params[part]
    return params


def _yaml_scalar(value) -> str:
    """Wert als YAML-Text, den PyYAML und DVC gleich lesen.

    PyYAML (YAML 1.1) liest 5e-05 als Text, nicht als Zahl; mit Punkt in der
    Mantisse (5.0e-05) ist es für beide eine Zahl.
    """
    text = repr(value) if isinstance(value, float) else str(value)
    if isinstance(value, float) and "e" in text and "." not in text.split("e")[0]:
        mantissa, exponent = text.split("e")
        text = f"{mantissa}.0e{exponent}"
    return text


def set_param(workdir: str | Path, key: str, value) -> None:
    """Ändert einen Wert in params.yaml der Arbeitskopie (Punktnotation, z. B. "gate.tolerance").

    Der Wert wird in der Zeile ersetzt, Kommentare bleiben stehen. DVC
    vergleicht Werte, nicht Text: Nur eine echte Änderung löst Stufen aus.
    """
    path = Path(workdir) / "params.yaml"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    parts, section, done = key.split("."), None, False
    for i, line in enumerate(lines):
        top = re.match(r"^(\w+):", line)
        if top:
            section = top.group(1)
        name = parts[-1]
        inside = (len(parts) == 1 and top) or (len(parts) == 2 and section == parts[0] and line.startswith("  "))
        match = re.match(rf"^(\s*{name}:\s*)([^#\n]*?)(\s*#.*)?$", line.rstrip("\n"))
        if inside and match:
            lines[i] = f"{match.group(1)}{_yaml_scalar(value)}{match.group(3) or ''}\n"
            done = True
            break
    if not done:
        raise KeyError(f"{key} nicht in {path}")
    path.write_text("".join(lines), encoding="utf-8")
    assert get_param(workdir, key) == value, f"{key} ließ sich nicht setzen"


def dvc(workdir: str | Path, *args: str) -> str:
    """Ruft DVC in der Arbeitskopie auf; `python` in dvc.yaml ist das Python dieser Umgebung."""
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
           "DVC_NO_ANALYTICS": "1"}
    done = subprocess.run([sys.executable, "-m", "dvc", *args], cwd=workdir, env=env,
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(done.stdout[-2000:] + done.stderr[-2000:])
    return done.stdout + done.stderr


def repro(workdir: str | Path) -> dict[str, str]:
    """dvc repro; liefert je Stufe "neu gerechnet" oder "übersprungen" und druckt die Kurzfassung.

    Ausgaben der Stufen (z. B. "gate: fail (...)") werden mitgedruckt.
    """
    out = dvc(workdir, "repro")
    status: dict[str, str] = {}
    for line in out.splitlines():
        ran = re.search(r"Running stage '(\w+)'", line)
        skipped = re.search(r"Stage '(\w+)' didn't change, skipping", line)
        restored = re.search(r"Stage '(\w+)' is cached", line)
        if ran:
            status[ran.group(1)] = "neu gerechnet"
        elif skipped:
            status[skipped.group(1)] = "übersprungen"
        elif restored:
            status[restored.group(1)] = "aus dem Run-Cache"
    for stage in STAGES:
        print(f"  {stage:14s} {status.get(stage, '?')}")
    for line in out.splitlines():
        if line.startswith(("train:", "evaluate:", "gate:")):
            print("  " + line[:140])
    return status


def ran(status: dict[str, str]) -> set[str]:
    """Namen der Stufen, die neu gerechnet haben."""
    return {stage for stage, state in status.items() if state == "neu gerechnet"}


def describe(workdir: str | Path) -> None:
    """Druckt je Stufe, wovon sie abhängt (Dateien und Parameter aus dvc.yaml)."""
    stages = yaml.safe_load((Path(workdir) / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
    for name, spec in stages.items():
        deps = ", ".join(d for d in spec.get("deps", []) if not d.endswith(".py"))
        params = ", ".join(spec.get("params", []))
        extra = "  (läuft immer)" if spec.get("always_changed") else ""
        print(f"  {name:14s} Dateien: {deps or '-':42s} Parameter: {params}{extra}")
