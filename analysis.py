"""
analysis.py
===========
Consistent publication-style figures for the shared-oracle studies.

The script keeps one visual system across every figure:
- identical typography, line weights, markers, and axis treatment;
- one concise figure heading;
- legends above the plotting area so they never cover data;
- smooth, linearly interpolated heatmaps;
- standard single- and double-column dimensions;
- matching PNG and vector PDF outputs.

Examples
--------
python analysis.py --study 0 --results results --out figures
python analysis.py --study all
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  Shared visual system
# --------------------------------------------------------------------------- #

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
    "er": "Erdős–Rényi",
    "ws": "Small-world",
    "ba": "Scale-free",
    "star": "Star",
}


def setup_style() -> None:
    """Set one restrained style for every output."""
    mpl.rcParams.update({
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
    })


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
    """Collect unique legend entries from one or more axes."""
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
    title: str,
    *,
    legend: bool = False,
    legend_ncol: int = 2,
) -> None:
    """
    Apply the same title and layout rules to every figure.

    Legends sit between the title and axes, never inside the data region.
    """
    axes_list = [axes] if isinstance(axes, plt.Axes) else list(np.ravel(axes))

    # fig.suptitle(
    #     title,
    #     x=0.015,
    #     y=0.985,
    #     ha="left",
    #     va="top",
    #     fontsize=9.4,
    #     fontweight="semibold",
    # )

    top = 0.79 if legend else 0.87
    # fig.tight_layout(rect=(0.0, 0.0, 1.0, top), pad=0.55, w_pad=0.85, h_pad=0.55)

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


def _save(fig: plt.Figure, out: str, name: str) -> None:
    os.makedirs(out, exist_ok=True)
    png = os.path.join(out, name)
    fig.savefig(png)
    fig.savefig(os.path.splitext(png)[0] + ".pdf")
    plt.close(fig)
    print(f"  wrote {png} (+ PDF)")


def _need(path: str) -> bool:
    if not os.path.exists(path):
        print(f"  [skip] missing {path}")
        return False
    return True


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _mean_ci(
    data: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Mean and approximate 95% Monte Carlo confidence interval."""
    selected = data[mask]
    if selected.shape[0] == 0:
        return None
    mean = selected.mean(axis=0)
    if selected.shape[0] == 1:
        ci = np.zeros_like(mean)
    else:
        ci = 1.96 * selected.std(axis=0, ddof=1) / np.sqrt(selected.shape[0])
    return mean, ci


def _plot_band(
    ax: plt.Axes,
    x: np.ndarray,
    data: np.ndarray,
    mask: np.ndarray,
    *,
    label: str,
    color: str,
) -> None:
    stats = _mean_ci(data, mask)
    if stats is None:
        return
    mean, ci = stats
    ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.14, linewidth=0)
    ax.plot(x, mean, color=color, label=label)


def _interpolate_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    nx: int = 240,
    ny: int = 180,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooth a rectangular field with piecewise-linear interpolation.

    This avoids blocky heatmaps without introducing a SciPy dependency.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)

    # Interpolate each row along x, then each column along y.
    zx = np.vstack([np.interp(xi, x, row) for row in z])
    zi = np.vstack([np.interp(yi, y, zx[:, j]) for j in range(zx.shape[1])]).T
    return xi, yi, zi


def smooth_heatmap(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    cmap: str,
    norm: Normalize,
):
    """Draw the same smooth heatmap style throughout the paper."""
    xi, yi, zi = _interpolate_grid(x, y, z)
    image = ax.imshow(
        zi,
        origin="lower",
        aspect="auto",
        extent=(xi.min(), xi.max(), yi.min(), yi.max()),
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",
        rasterized=True,
    )
    return image, xi, yi, zi


def style_heatmap_axes(ax: plt.Axes) -> None:
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


# --------------------------------------------------------------------------- #
#  Study 0
# --------------------------------------------------------------------------- #

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
        subset = (
            data[data["epsilon"] == epsilon]
            .set_index("topology")
            .reindex(topologies)
        )
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
    finish_figure(
        fig,
        ax,
        "Baseline reliability",
        legend=True,
        legend_ncol=len(epsilons),
    )
    _save(fig, out, "study0_reliability.png")


# --------------------------------------------------------------------------- #
#  Study 1
# --------------------------------------------------------------------------- #

def fig_study1(results: str, out: str) -> None:
    grid_path = os.path.join(results, "study1_grid.csv")
    shared_path = os.path.join(results, "study1_sharedvsindep.csv")
    isolated_path = os.path.join(results, "study1_isolated.csv")

    # Reliability heatmaps.
    if _need(grid_path):
        df = pd.read_csv(grid_path)
        topologies = [t for t in ["complete", "cycle"] if t in set(df["topology"])]

        fig, axes = plt.subplots(
            1,
            len(topologies),
            figsize=DOUBLE,
            sharex=True,
            sharey=True,
        )
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
                cmap="viridis",
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

        finish_figure(fig, axes, "Frozen oracle")
        _save(fig, out, "study1_heatmaps.png")

    # Shared versus independent signals.
    if _need(shared_path):
        df = pd.read_csv(shared_path).copy()
        df["_shared"] = _as_bool(df["shared"])
        topologies = [t for t in ["complete", "cycle"] if t in set(df["topology"])]

        fig, axes = plt.subplots(
            1,
            len(topologies),
            figsize=DOUBLE,
            sharex=True,
            sharey=True,
        )
        axes = np.atleast_1d(axes)

        for ax, topology in zip(axes, topologies):
            shared = df[
                (df["topology"] == topology) & df["_shared"]
            ].sort_values("oracle_r")
            independent = df[
                (df["topology"] == topology) & ~df["_shared"]
            ].sort_values("oracle_r")

            ax.plot(
                independent["oracle_r"],
                independent["reliability"],
                color=BLUE,
                marker="o",
                markerfacecolor="white",
                label="Independent",
            )
            ax.plot(
                shared["oracle_r"],
                shared["reliability"],
                color=ORANGE,
                marker="o",
                markerfacecolor="white",
                label="Shared",
            )
            ax.axvline(0.5, color=GRAY, linewidth=0.75, linestyle=":")
            ax.set_title(TOPO_NAME[topology])
            ax.set_xlabel(r"Oracle reliability $r$")
            ax.set_ylim(0, 1.1)
            clean_axes(ax)

        axes[0].set_ylabel("Correct consensus")
        finish_figure(
            fig,
            axes,
            "Shared vs. independent",
            legend=True,
            legend_ncol=2,
        )
        _save(fig, out, "study1_shared_vs_independent.png")

    # Community and individual performance.
    if _need(grid_path):
        df = pd.read_csv(grid_path)
        isolated = pd.read_csv(isolated_path) if os.path.exists(isolated_path) else None
        topologies = [t for t in ["complete", "cycle"] if t in set(df["topology"])]

        fig, axes = plt.subplots(
            1,
            len(topologies),
            figsize=DOUBLE,
            sharex=True,
            sharey=True,
        )
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

            if "se" in full.columns:
                se = full["se"].to_numpy()
                ax.fill_between(
                    r,
                    reliability - se,
                    reliability + se,
                    color=BLUE,
                    alpha=0.12,
                    linewidth=0,
                )

            ax.plot(r, reliability, color=BLUE, label="Community")
            ax.axhline(
                baseline,
                color=GRAY,
                linewidth=0.9,
                linestyle="--",
                label="No oracle",
            )

            if isolated is not None:
                iso = isolated.sort_values("oracle_r")
                ax.plot(
                    iso["oracle_r"],
                    iso["isolated_reliability"],
                    color=ORANGE,
                    linestyle=":",
                    label="Single",
                )

            ax.axvline(0.5, color=GRAY, linewidth=0.75, linestyle=":")
            ax.set_title(TOPO_NAME[topology])
            ax.set_xlabel(r"Oracle reliability $r$")
            ax.set_ylim(0, 1)
            clean_axes(ax)

        axes[0].set_ylabel("Correct consensus")
        finish_figure(
            fig,
            axes,
            "Community and individual",
            legend=True,
            legend_ncol=3,
        )
        _save(fig, out, "study1_phase_and_band.png")


# --------------------------------------------------------------------------- #
#  Study 2
# --------------------------------------------------------------------------- #

def fig_study2(results: str, out: str) -> None:
    reliability_path = os.path.join(results, "study2_reliability.csv")

    if _need(reliability_path):
        df = pd.read_csv(reliability_path)
        topologies = [t for t in ["complete", "cycle"] if t in set(df["topology"])]
        order = ["none", "frozen0.7", "live"]
        labels = ["None", "Frozen", "Live"]

        x = np.arange(len(order))
        width = 0.34

        fig, ax = plt.subplots(figsize=SINGLE)

        for i, topology in enumerate(topologies):
            subset = df[df["topology"] == topology]
            if "adoption_fraction" in subset.columns:
                subset = subset[
                    subset["adoption_fraction"] == subset["adoption_fraction"].max()
                ]
            values = [
                subset[subset["oracle"] == oracle]["reliability"].mean()
                for oracle in order
            ]
            ax.bar(
                x + (i - (len(topologies) - 1) / 2) * width,
                values,
                width,
                color=TOPO_COLOR[topology],
                label=TOPO_NAME[topology],
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Correct consensus")
        ax.set_ylim(0, 1)
        clean_axes(ax)
        finish_figure(
            fig,
            ax,
            "Frozen vs. live",
            legend=True,
            legend_ncol=2,
        )
        _save(fig, out, "study2_reliability.png")

    timeseries_path = os.path.join(results, "study2_timeseries.npz")
    if not _need(timeseries_path):
        return

    data = np.load(timeseries_path)
    correct = data["correct"].astype(bool)
    rounds = np.arange(data["entropy"].shape[1])

    # Exploration entropy.
    fig, ax = plt.subplots(figsize=SINGLE)
    _plot_band(ax, rounds, data["entropy"], correct, label="Correct", color=BLUE)
    _plot_band(ax, rounds, data["entropy"], ~correct, label="Incorrect", color=ORANGE)
    ax.set_xlabel("Round")
    ax.set_ylabel("Entropy")
    ax.set_xlim(rounds[0], rounds[-1])
    ax.set_ylim(0, 1)
    clean_axes(ax)
    finish_figure(
        fig,
        ax,
        "Exploration entropy",
        legend=True,
        legend_ncol=2,
    )
    _save(fig, out, "study2_entropy_collapse.png")

    # Fraction testing the better action.
    fig, ax = plt.subplots(figsize=SINGLE)
    _plot_band(ax, rounds, data["frac_B"], correct, label="Correct", color=BLUE)
    _plot_band(ax, rounds, data["frac_B"], ~correct, label="Incorrect", color=ORANGE)
    ax.set_xlabel("Round")
    ax.set_ylabel(r"Fraction testing $b$")
    ax.set_xlim(rounds[0], rounds[-1])
    ax.set_ylim(0, 1.1)
    clean_axes(ax)
    finish_figure(
        fig,
        ax,
        "Testing the better arm",
        legend=True,
        legend_ncol=2,
    )
    _save(fig, out, "study2_exploration.png")

    # Live-oracle belief.
    fig, ax = plt.subplots(figsize=SINGLE)
    _plot_band(ax, rounds, data["oracle_value"], correct, label="Correct", color=BLUE)
    _plot_band(ax, rounds, data["oracle_value"], ~correct, label="Incorrect", color=ORANGE)
    ax.axhline(0.50, color=GRAY, linewidth=0.8, linestyle="--", label=r"$p_s$")
    ax.axhline(0.55, color=GRAY, linewidth=0.8, linestyle=":", label=r"$p_b$")
    ax.set_xlabel("Round")
    ax.set_ylabel("Oracle belief")
    ax.set_xlim(rounds[0], rounds[-1])
    ax.set_ylim(0.25, 0.75)
    clean_axes(ax)
    finish_figure(
        fig,
        ax,
        "Oracle belief",
        legend=True,
        legend_ncol=4,
    )
    _save(fig, out, "study2_oracle_freeze.png")


# --------------------------------------------------------------------------- #
#  Study 3
# --------------------------------------------------------------------------- #

def fig_study3(results: str, out: str) -> None:
    endogenous_path = os.path.join(results, "study3_endogenous.csv")
    grid_path = os.path.join(results, "study1_grid.csv")
    robustness_path = os.path.join(results, "study3_robustness.csv")
    trajectories_path = os.path.join(results, "study3_trust_traj.npz")

    # Network inversion.
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
        finish_figure(
            fig,
            ax,
            "Network inversion",
            legend=True,
            legend_ncol=2,
        )
        _save(fig, out, "study3_network_inversion.png")

    # Fixed versus adaptive trust.
    if _need(endogenous_path):
        df = pd.read_csv(endogenous_path).copy()
        df["_adaptive"] = _as_bool(df["endogenous_trust"])
        topologies = [t for t in ["complete", "cycle"] if t in set(df["topology"])]

        fig, axes = plt.subplots(
            1,
            len(topologies),
            figsize=DOUBLE,
            sharex=True,
            sharey=True,
        )
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
        finish_figure(
            fig,
            axes,
            "Adaptive trust",
            legend=True,
            legend_ncol=2,
        )
        _save(fig, out, "study3_endogenous_trust.png")

    # Trust trajectories.
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
        finish_figure(
            fig,
            ax,
            "Trust trajectories",
            legend=True,
            legend_ncol=2,
        )
        _save(fig, out, "study3_trust_trajectories.png")

    # Correlation-penalty robustness.
    if _need(robustness_path):
        df = pd.read_csv(robustness_path)
        n0 = sorted(df["n_agents"].unique())[0]
        data = df[df["n_agents"] == n0]

        pivot = (
            data.pivot_table(
                index="oracle_tau",
                columns="epsilon",
                values="penalty",
            )
            .sort_index()
            .sort_index(axis=1)
        )

        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            print("  [skip] robustness heatmap requires a two-dimensional sweep")
        else:
            x = pivot.columns.to_numpy(dtype=float)
            y = pivot.index.to_numpy(dtype=float)
            z = pivot.to_numpy(dtype=float)

            vmax = float(np.nanmax(np.abs(z))) or 0.1
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

            fig, ax = plt.subplots(figsize=SINGLE)
            image, _, _, _ = smooth_heatmap(
                ax,
                x,
                y,
                z,
                cmap="RdBu_r",
                norm=norm,
            )
            ax.set_xlabel(r"Arm gap $\varepsilon$")
            ax.set_ylabel(r"Trust $\tau$")
            style_heatmap_axes(ax)

            colorbar = fig.colorbar(image, ax=ax, shrink=0.83, pad=0.025)
            colorbar.set_label("Independent − shared")
            colorbar.outline.set_visible(False)

            finish_figure(fig, ax, "Correlation penalty")
            _save(fig, out, "study3_robustness.png")


# --------------------------------------------------------------------------- #
#  Command line
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper figures.")
    parser.add_argument(
        "--study",
        default="all",
        choices=["0", "1", "2", "3", "all"],
    )
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    setup_style()

    functions = {
        "0": fig_study0,
        "1": fig_study1,
        "2": fig_study2,
        "3": fig_study3,
    }
    studies = ["0", "1", "2", "3"] if args.study == "all" else [args.study]

    for study in studies:
        print(f"Study {study}")
        functions[study](args.results, args.out)


if __name__ == "__main__":
    main()

