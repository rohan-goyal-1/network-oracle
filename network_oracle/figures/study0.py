"""Figures for Study 0: baseline network reliability."""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from network_oracle.figures.style import DOUBLE, GRAY, TOPO_NAME, clean_axes, finish_figure
from network_oracle.figures.utils import _need, _save


def fig_study0(results: str, out: str) -> None:
    path = os.path.join(results, "study0.csv")
    if not _need(path):
        return

    df = pd.read_csv(path)
    n0 = sorted(df["n_agents"].unique())[0]
    data = df[df["n_agents"] == n0]

    order = ["cycle", "wheel", "ws", "ba", "er", "complete"]
    topologies = [t for t in order if t in set(data["topology"])]
    epsilons = sorted(data["epsilon"].unique())

    cmap = mpl.colormaps["Blues"]
    shades = [cmap(v) for v in np.linspace(0.42, 0.82, len(epsilons))]

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
            error_kw={"ecolor": GRAY, "elinewidth": 0.7, "capsize": 1.8},
            label=fr"$\varepsilon={epsilon:g}$",
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([TOPO_NAME[t] for t in topologies])
    ax.set_ylabel("Correct consensus")
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.linspace(0, 1, 6))
    clean_axes(ax)
    finish_figure(fig, ax, "Baseline reliability", legend=True, legend_ncol=len(epsilons))
    _save(fig, out, "study0_reliability.png")
