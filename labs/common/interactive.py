"""Bausteine für Demos und Übungen: erst tippen, dann messen; Regler für Parameter.

Zwei Muster tragen die interaktiven Teile der Labs:

- `check_guess`: Die Teilnehmenden legen sich vor dem Ausführen auf einen Wert
  fest (Variable mit einem Tipp), danach vergleicht die Zelle Tipp und Messwert.
- `explore`: Regler oder Auswahllisten für einen Parameter, dessen Wirkung die
  eigentliche Lektion ist (Drift-Anteil, Batchgröße, max_seq_length).

Beim Bau der Notebooks (scripts/build_lab.py setzt MLOPS_LAB_BUILD=1), ohne
ipywidgets und mit MLOPS_WIDGETS=0 ruft `explore` die Funktion einmal mit den Startwerten auf. So steht
in der Trainerfassung eine statische Ausgabe, und das Notebook läuft auch in
Umgebungen ohne Widget-Unterstützung.
"""
from __future__ import annotations

import os
from typing import Any, Callable


def building() -> bool:
    """True, während scripts/build_lab.py das Notebook ausführt."""
    return os.environ.get("MLOPS_LAB_BUILD") == "1"


def widgets_disabled() -> bool:
    """True, wenn Regler abgeschaltet sind: MLOPS_WIDGETS=0 (z. B. in Editoren ohne Widget-Anzeige wie Zed)."""
    return os.environ.get("MLOPS_WIDGETS") == "0"


def check_guess(
    name: str,
    guess: Any,
    measured: Any,
    tolerance: float | None = None,
    fmt: str = "{:.3f}",
) -> bool | None:
    """Vergleicht einen Tipp mit dem Messwert und druckt das Ergebnis.

    Zahlen: Treffer, wenn |Tipp - Messwert| <= tolerance (ohne tolerance nur
    die Abweichung). Texte und andere Werte: Treffer bei Gleichheit.
    Liefert True/False, ohne Tipp None.
    """
    def show(value: Any) -> str:
        return fmt.format(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else repr(value)

    if guess is None:
        print(f"{name}: kein Tipp abgegeben. Gemessen: {show(measured)}")
        return None
    if isinstance(guess, (int, float)) and isinstance(measured, (int, float)):
        diff = float(measured) - float(guess)
        line = f"{name}: Tipp {show(guess)}, gemessen {show(measured)} (Abweichung {diff:+.3g})"
        if tolerance is None:
            print(line)
            return None
        hit = abs(diff) <= tolerance
        print(line + (" Treffer." if hit else f" daneben (Toleranz ± {tolerance:g})."))
        return hit
    hit = guess == measured
    print(f"{name}: Tipp {show(guess)}, gemessen {show(measured)}" + (" Treffer." if hit else " daneben."))
    return hit


def _widget(name: str, spec: Any):
    import ipywidgets as w

    if isinstance(spec, list):
        return w.Dropdown(options=spec, value=spec[0], description=name)
    start, lo, hi, step = spec
    cls = w.IntSlider if all(isinstance(v, int) for v in spec) else w.FloatSlider
    # continuous_update=False: erst beim Loslassen rechnen, sonst stauen sich teure Aufrufe
    return cls(value=start, min=lo, max=hi, step=step, description=name, continuous_update=False)


def _start(spec: Any) -> Any:
    return spec[0]


def explore(fn: Callable[..., Any], manual: bool = False, **controls: Any) -> Any:
    """Zeigt Regler für fn(**werte) und ruft fn bei jeder Änderung neu auf.

    Jeder Regler ist entweder eine Liste von Optionen (Startwert = erste Option)
    oder ein Tupel (start, min, max, step). manual=True zeigt zusätzlich einen
    Knopf "Ausführen" und rechnet erst auf Klick (für Aufrufe über einer Sekunde).
    Beim Bau und ohne ipywidgets: ein Aufruf mit den Startwerten.
    """
    starts = {k: _start(v) for k, v in controls.items()}
    try:
        import ipywidgets as w
    except ImportError:
        w = None
    if w is None or building() or widgets_disabled():
        shown = ", ".join(f"{k}={v!r}" for k, v in starts.items())
        print(f"Statische Fassung mit den Startwerten ({shown}). Im laufenden Notebook stehen hier Regler.")
        return fn(**starts)
    widgets = {k: _widget(k, v) for k, v in controls.items()}
    if manual:
        return w.interact_manual.options(manual_name="Ausführen")(fn, **widgets)
    return w.interact(fn, **widgets)
