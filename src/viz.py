"""Shared plotting style.

Colours come from a validated categorical palette: adjacent and all-pairs
separation checked for normal vision and for deuteranopia/tritanopia, so the
figures stay readable for colour-blind readers and in greyscale print.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdcd8"

# Categorical slots, fixed order, never cycled.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "font.size": 9.5,
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def caption(fig, text: str) -> None:
    """One line under the figure saying what the reader should take away."""
    fig.text(0.0, -0.02, text, ha="left", va="top",
             fontsize=8.5, color=INK_MUTED, wrap=True)
