"""
Plot style and figure saving. All labels in English (True Stress, Strain Rate, True Strain).
"""

import matplotlib.pyplot as plt
import os
import re
from typing import Optional

METHOD_COLORS = {
    "Arrhenius": "#1f77b4",
    "NN-Direct": "#ff7f0e",
    "NN-Arrhenius-Only": "#7f7f7f",
    "NN-PhysicsInit": "#d62728",
}
METHOD_STYLES = {
    "Arrhenius": "--",
    "NN-Direct": "-.",
    "NN-Arrhenius-Only": ":",
    "NN-PhysicsInit": "-",
}
METHOD_ORDER = ["Arrhenius", "NN-Direct", "NN-Arrhenius-Only", "NN-PhysicsInit"]


def setup_style() -> None:
    """Set matplotlib rcParams for publication-style figures (English, no Chinese)."""
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def _safe_figure_name(name: str) -> str:
    name = str(name)
    name = re.sub(r"[^A-Za-z0-9._\-+()\[\]{} ]+", " ", name)
    name = re.sub(r"\s+", "_", name).strip("_")
    return name if name else "figure"


def save_fig(fig, name: str, output_dir: str) -> str:
    """Save figure to output_dir with a safe filename. Returns path."""
    os.makedirs(output_dir, exist_ok=True)
    safe = _safe_figure_name(name)
    path = os.path.join(output_dir, f"{safe}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    return path
