"""Shared figure style for docs/scripts/*.py (not a figure itself).

Each figure script imports this module, draws one figure and saves it as
``docs/img/<script name>.png`` with :func:`save`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IMG_DIR = Path(__file__).resolve().parents[1] / "img"
GENERATED_DIR = Path(__file__).resolve().parents[1] / "generated"

# Analytic or true answer in black, package output in blue, a second series in
# orange, approximations in grey, warnings in red.
TRUTH = "black"
PACKAGE = "#1f77b4"
SECOND = "#ff7f0e"
APPROX = "0.55"
HIGHLIGHT = "#d62728"
SERIES = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b")

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def save(fig, script_file: str) -> Path:
    """Save ``fig`` as docs/img/<stem of script_file>.png and close it."""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    path = IMG_DIR / (Path(script_file).stem + ".png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return path


def panel_label(ax, text: str) -> None:
    """Bold panel letter in the upper-left corner of ``ax``."""
    ax.text(
        0.02,
        0.97,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1},
    )
