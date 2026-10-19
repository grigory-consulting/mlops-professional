"""Einheitlicher Plot-Stil der Labs und Folien (Palette Seminarblau)."""
from __future__ import annotations

import matplotlib as mpl

PAPER = "#f4f7fb"
INK = "#183b66"
ACCENT = "#2459a6"
LIGHT = "#66a3ff"
MUTED = "#5a6b82"
WARN = "#b45309"
GRID = "#b9c6d8"
SERIES = [ACCENT, WARN, LIGHT, INK, MUTED]


def use_course_style() -> None:
    """Setzt Farben, Schrift und Raster für alle folgenden matplotlib-Plots."""
    mpl.rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "font.size": 12,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
