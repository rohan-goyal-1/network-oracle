"""Figures for Study 3: biased oracle, adaptive trust, and robustness."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from network_oracle.figures.style import (
    BLUE,
    DOUBLE,
    GRAY,
    GREEN,
    ORANGE,
    SINGLE,
    TOPO_COLOR,
    TOPO_NAME,
    clean_axes,
    finish_figure,
)
from network_oracle.figures.utils import (
    _as_bool,
    _need,
    _save,
    smooth_heatmap,
    style_heatmap_axes,
)


def fig_study3(results: str, out: str) -> None:
    endogenous_path = os.path.join(results, "study3_endogenous.csv")
    grid_path = os.path.join(results, "study1_grid.csv")
    robustness_path = os.path.join(results, "study3_robustness.csv")
    trajectories_path = os.path.join(results, "study3_trust_traj.npz")

    if _need(grid_path):
        df = pd.read_csv(grid_path)
        full = df[np.isclose(df["adoption_fraction"], 1.0)]
        topologies = [t for t in ["complete", "cycle"] if t in set(full["topology"])]

        fig, ax = plt.subplots(figsize=SINGLE)
        for topology in topologies:
            subset = full[full["topology"] == topology].sort_values("oracle_r")
            ax.plot(
                subset["oracle_r"],
                subset["reliability"],
                color=TOPO_COLOR[topology],
                marker="o",
                markerfacecolor="white",
                label=TOPO_NAME[topology],
            )

        ax.axvline(0.5, color=GRAY, linewidth=0.75, linestyle=":")
        ax.set_xlabel(r"Oracle reliability $r$")
        ax.set_ylabel("Correct consensus")
        ax.set_ylim(0, 1.1)
        clean_axes(ax)
        finish_figure(fig, ax, "Network inversion", legend=True, legend_ncol=2)
        _save(fig, out, "study3_network_inversion.png")

    if _need(endogenous_path):
        df = pd.read_csv(endogenous_path).copy()
        df["_adaptive"] = _as_bool(df["endogenous_trust"])
        topologies = [t for t in ["complete", "cycle"] if t in set(df["topology"])]

        fig, axes = plt.subplots(1, len(topologies), figsize=DOUBLE, sharex=True, sharey=True)
        axes = np.atleast_1d(axes)

        for ax, topology in zip(axes, topologies):
            subset = df[df["topology"] == topology]
            for adaptive, color, label, linestyle in [
                (False, GRAY, "Fixed", "--"),
                (True, BLUE, "Adaptive", "-"),
            ]:
                curve = (
                    subset[subset["_adaptive"] == adaptive]
                    .groupby("oracle_r", as_index=False)["reliability"]
                    .mean()
                    .sort_values("oracle_r")
                )
                ax.plot(
                    curve["oracle_r"],
                    curve["reliability"],
                    color=color,
                    linestyle=linestyle,
                    marker="o",
                    markerfacecolor="white",
                    label=label,
                )

            ax.axvline(0.5, color=GRAY, linewidth=0.75, linestyle=":")
            ax.set_title(TOPO_NAME[topology])
            ax.set_xlabel(r"Oracle reliability $r$")
            ax.set_ylim(0, 1.1)
            clean_axes(ax)

        axes[0].set_ylabel("Correct consensus")
        finish_figure(fig, axes, "Adaptive trust", legend=True, legend_ncol=2)
        _save(fig, out, "study3_endogenous_trust.png")

    if _need(trajectories_path):
        data = np.load(trajectories_path)
        labels = {
            "biased_r0.3": ("Biased", ORANGE),
            "reliable_r0.8": ("Reliable", BLUE),
        }

        fig, ax = plt.subplots(figsize=SINGLE)
        for key in data.files:
            label, color = labels.get(key, (key, GREEN))
            values = data[key]
            ax.plot(np.arange(len(values)), values, color=color, label=label)

        ax.set_xlabel("Round")
        ax.set_ylabel("Mean trust")
        clean_axes(ax)
        finish_figure(fig, ax, "Trust trajectories", legend=True, legend_ncol=2)
        _save(fig, out, "study3_trust_trajectories.png")

    if _need(robustness_path):
        df = pd.read_csv(robustness_path)
        n0 = sorted(df["n_agents"].unique())[0]
        data = df[df["n_agents"] == n0]

        pivot = (
            data.pivot_table(index="oracle_tau", columns="epsilon", values="penalty")
            .sort_index()
            .sort_index(axis=1)
        )

        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            print("  [skip] robustness heatmap requires a two-dimensional sweep")
            return

        x = pivot.columns.to_numpy(dtype=float)
        y = pivot.index.to_numpy(dtype=float)
        z = pivot.to_numpy(dtype=float)
        vmax = float(np.nanmax(np.abs(z))) or 0.1
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        fig, ax = plt.subplots(figsize=SINGLE)
        image, _, _, _ = smooth_heatmap(ax, x, y, z, cmap="RdBu_r", norm=norm)
        ax.set_xlabel(r"Arm gap $\varepsilon$")
        ax.set_ylabel(r"Trust $\tau$")
        style_heatmap_axes(ax)

        colorbar = fig.colorbar(image, ax=ax, shrink=0.83, pad=0.025)
        colorbar.set_label("Independent - shared")
        colorbar.outline.set_visible(False)

        finish_figure(fig, ax, "Correlation penalty")
        _save(fig, out, "study3_robustness.png")
