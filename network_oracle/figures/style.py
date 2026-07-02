"""Shared visual system for paper figures."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SINGLE = (3.40, 2.55)
DOUBLE = (7.05, 3.00)

BLUE = "#3B6FB6"
ORANGE = "#D97732"
GREEN = "#4C956C"
GRAY = "#666666"
LIGHT_GRAY = "#E5E5E5"

TOPO_COLOR = {
    "complete": BLUE,
    "cycle": ORANGE,
    "wheel": GREEN,
    "er": "#A05A9C",
    "ws": "#7B8F35",
    "ba": "#4E9CB5",
    "star": "#8B6D5C",
}

TOPO_NAME = {
    "complete": "Complete",
    "cycle": "Cycle",
    "wheel": "Wheel",
    "er": "Erdos-Renyi",
    "ws": "Small-world",
    "ba": "Scale-free",
    "star": "Star",
}


def setup_style() -> None:
    """Set one restrained style for every output."""
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 8.0,
            "axes.titlesize": 8.4,
            "axes.titleweight": "semibold",
            "axes.titlepad": 4.0,
            "axes.labelsize": 8.0,
            "axes.labelpad": 3.0,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.fontsize": 7.1,
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.1,
            "lines.linewidth": 1.5,
            "lines.markersize": 3.2,
            "lines.markeredgewidth": 0.7,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0.5,
        }
    )


def clean_axes(ax: plt.Axes, *, grid: bool = True) -> None:
    """Use the same quiet axis treatment everywhere."""
    ax.spines["left"].set_color(GRAY)
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(color=GRAY)
    if grid:
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.5)
    else:
        ax.grid(False)


def _unique_legend(axes: Iterable[plt.Axes]) -> tuple[list, list[str]]:
    entries: dict[str, object] = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and not label.startswith("_") and label not in entries:
                entries[label] = handle
    return list(entries.values()), list(entries.keys())


def finish_figure(
    fig: plt.Figure,
    axes: plt.Axes | Iterable[plt.Axes],
    _title: str,
    *,
    legend: bool = False,
    legend_ncol: int = 2,
) -> None:
    """Apply shared legend placement rules to a figure."""
    axes_list = [axes] if isinstance(axes, plt.Axes) else list(np.ravel(axes))
    if legend:
        handles, labels = _unique_legend(axes_list)
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.930),
                ncol=min(legend_ncol, len(handles)),
                borderaxespad=0.0,
            )
