"""Figures for baseline network reliability."""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from network_oracle.figures.style import (
    DOUBLE,
    RELIABILITY_CMAP,
    TOPO_NAME,
    clean_axes,
    errorbar_style,
)
from network_oracle.figures.utils import _need, _save


def figure(results: str, out: str) -> None:
    path = os.path.join(results, "baseline_reliability.csv")
    if not _need(path):
        return

    df = pd.read_csv(path)
    n0 = sorted(df["n_agents"].unique())[0]
    data = df[df["n_agents"] == n0]

    order = ["cycle", "wheel", "ws", "ba", "er", "complete"]
    topologies = [t for t in order if t in set(data["topology"])]
    epsilons = sorted(data["epsilon"].unique())

    cmap = mpl.colormaps[RELIABILITY_CMAP]
    norm = Normalize(vmin=min(epsilons), vmax=max(epsilons))
    shades = [cmap(norm(epsilon)) for epsilon in epsilons]

    x = np.arange(len(topologies))
    width = 0.78 / max(len(epsilons), 1)

    fig, ax = plt.subplots(figsize=DOUBLE)
    for i, epsilon in enumerate(epsilons):
        subset = data[data["epsilon"] == epsilon].set_index("topology").reindex(topologies)
        xpos = x + (i - (len(epsilons) - 1) / 2) * width
        ax.bar(
            xpos,
            subset["reliability"],
            width,
            yerr=subset["se"],
            color=shades[i],
            edgecolor="white",
            error_kw=errorbar_style(),
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([TOPO_NAME[t] for t in topologies])
    ax.set_ylabel("Correct consensus")
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.linspace(0, 1, 6))
    clean_axes(ax)

    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        orientation="horizontal",
        location="top",
        fraction=0.055,
        pad=0.08,
        aspect=28,
    )
    colorbar.set_label(r"Arm gap $\varepsilon$")
    colorbar.set_ticks(epsilons)
    colorbar.outline.set_visible(False)

    _save(fig, out, "baseline_network_reliability.png")
