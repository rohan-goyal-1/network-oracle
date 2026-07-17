"""Figures for fixed shared-testimony sweeps."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from network_oracle.figures.style import (
    BLUE,
    DOUBLE,
    GRAY,
    ORANGE,
    RELIABILITY_CMAP,
    TOPO_NAME,
    clean_axes,
    finish_figure,
    reference_line_style,
)
from network_oracle.figures.utils import (
    _as_bool,
    _need,
    _save,
    plot_curve,
    smooth_heatmap,
    style_heatmap_axes,
)


def figure(results: str, out: str) -> None:
    grid_path = os.path.join(results, "fixed_testimony_grid.csv")
    shared_path = os.path.join(results, "shared_vs_independent_testimony.csv")
    isolated_path = os.path.join(results, "single_agent_testimony.csv")

    if _need(grid_path):
        df = pd.read_csv(grid_path)
        topologies = [t for t in ["complete", "cycle"] if t in set(df["topology"])]

        fig, axes = plt.subplots(1, len(topologies), figsize=DOUBLE, sharex=True, sharey=True)
        axes = np.atleast_1d(axes)
        image = None

        for ax, topology in zip(axes, topologies):
            pivot = (
                df[df["topology"] == topology]
                .pivot_table(
                    index="adoption_fraction",
                    columns="oracle_r",
                    values="reliability",
                )
                .sort_index()
                .sort_index(axis=1)
            )
            x = pivot.columns.to_numpy(dtype=float)
            y = pivot.index.to_numpy(dtype=float)
            z = pivot.to_numpy(dtype=float)

            image, xi, yi, zi = smooth_heatmap(
                ax,
                x,
                y,
                z,
                cmap=RELIABILITY_CMAP,
                norm=Normalize(vmin=0, vmax=1),
            )

            baseline = df[
                (df["topology"] == topology)
                & np.isclose(df["adoption_fraction"], 0.0)
            ]["reliability"].mean()
            if np.isfinite(baseline):
                xx, yy = np.meshgrid(xi, yi)
                ax.contour(
                    xx,
                    yy,
                    zi,
                    levels=[baseline],
                    colors="white",
                    linewidths=0.85,
                    linestyles=["--"],
                )

            ax.set_title(TOPO_NAME[topology])
            ax.set_xlabel(r"Oracle reliability $r$")
            style_heatmap_axes(ax)

        axes[0].set_ylabel(r"Adoption $\varphi$")
        colorbar = fig.colorbar(image, ax=axes, shrink=0.84, pad=0.018)
        colorbar.set_label("Correct consensus")
        colorbar.outline.set_visible(False)
        finish_figure(fig, axes, "Fixed testimony")
        _save(fig, out, "fixed_testimony_heatmaps.png")

    if _need(shared_path):
        df = pd.read_csv(shared_path).copy()
        df["_shared"] = _as_bool(df["shared"])
        topologies = [t for t in ["complete", "cycle"] if t in set(df["topology"])]

        fig, axes = plt.subplots(1, len(topologies), figsize=DOUBLE, sharex=True, sharey=True)
        axes = np.atleast_1d(axes)

        for ax, topology in zip(axes, topologies):
            shared = df[(df["topology"] == topology) & df["_shared"]].sort_values("oracle_r")
            independent = df[(df["topology"] == topology) & ~df["_shared"]].sort_values(
                "oracle_r"
            )

            plot_curve(
                ax,
                independent["oracle_r"],
                independent["reliability"],
                color=BLUE,
                label="Independent",
                se=independent["se"] if "se" in independent.columns else None,
            )
            plot_curve(
                ax,
                shared["oracle_r"],
                shared["reliability"],
                color=ORANGE,
                label="Shared",
                se=shared["se"] if "se" in shared.columns else None,
            )
            ax.axvline(0.5, **reference_line_style())
            ax.set_title(TOPO_NAME[topology])
            ax.set_xlabel(r"Oracle reliability $r$")
            ax.set_ylim(0, 1.1)
            clean_axes(ax)

        axes[0].set_ylabel("Correct consensus")
        finish_figure(fig, axes, "Shared vs. independent", legend=True, legend_ncol=2)
        _save(fig, out, "shared_vs_independent_testimony.png")

    if _need(grid_path):
        df = pd.read_csv(grid_path)
        isolated = pd.read_csv(isolated_path) if os.path.exists(isolated_path) else None
        topologies = [t for t in ["complete", "cycle"] if t in set(df["topology"])]

        fig, axes = plt.subplots(1, len(topologies), figsize=DOUBLE, sharex=True, sharey=True)
        axes = np.atleast_1d(axes)

        for ax, topology in zip(axes, topologies):
            full = df[
                (df["topology"] == topology)
                & np.isclose(df["adoption_fraction"], 1.0)
            ].sort_values("oracle_r")
            baseline = df[
                (df["topology"] == topology)
                & np.isclose(df["adoption_fraction"], 0.0)
            ]["reliability"].mean()

            r = full["oracle_r"].to_numpy()
            reliability = full["reliability"].to_numpy()
            plot_curve(
                ax,
                r,
                reliability,
                color=BLUE,
                label="Community",
                se=full["se"] if "se" in full.columns else None,
            )
            ax.axhline(baseline, label="No oracle", **reference_line_style(linestyle="--"))

            if isolated is not None:
                iso = isolated.sort_values("oracle_r")
                plot_curve(
                    ax,
                    iso["oracle_r"],
                    iso["isolated_reliability"],
                    color=ORANGE,
                    label="Single",
                    se=iso["isolated_se"] if "isolated_se" in iso.columns else None,
                    linestyle=":",
                )

            ax.axvline(0.5, **reference_line_style())
            ax.set_title(TOPO_NAME[topology])
            ax.set_xlabel(r"Oracle reliability $r$")
            ax.set_ylim(0, 1.1)
            clean_axes(ax)

        axes[0].set_ylabel("Correct consensus")
        finish_figure(fig, axes, "Community and individual", legend=True, legend_ncol=3)
        _save(fig, out, "individual_help_collective_harm.png")
