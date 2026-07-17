"""Figures for adaptive oracle feedback."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from network_oracle.figures.style import (
    BLUE,
    ORANGE,
    SINGLE,
    TOPO_COLOR,
    TOPO_NAME,
    clean_axes,
    errorbar_style,
    finish_figure,
    reference_line_style,
)
from network_oracle.figures.utils import _need, _plot_band, _save


def figure(results: str, out: str) -> None:
    reliability_path = os.path.join(results, "adaptive_feedback_reliability.csv")

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
                subset = subset[subset["adoption_fraction"] == subset["adoption_fraction"].max()]
            values = [subset[subset["oracle"] == oracle]["reliability"].mean() for oracle in order]
            errors = [
                subset[subset["oracle"] == oracle]["se"].mean()
                if "se" in subset.columns and not subset[subset["oracle"] == oracle].empty
                else 0.0
                for oracle in order
            ]
            ax.bar(
                x + (i - (len(topologies) - 1) / 2) * width,
                values,
                width,
                yerr=errors,
                color=TOPO_COLOR[topology],
                edgecolor="white",
                error_kw=errorbar_style(),
                label=TOPO_NAME[topology],
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Correct consensus")
        ax.set_ylim(0, 1)
        clean_axes(ax)
        finish_figure(fig, ax, "Fixed vs. adaptive", legend=True, legend_ncol=2)
        _save(fig, out, "adaptive_feedback_reliability.png")

    timeseries_path = os.path.join(results, "adaptive_feedback_timeseries.npz")
    if not _need(timeseries_path):
        return

    data = np.load(timeseries_path)
    correct = data["correct"].astype(bool)
    rounds = np.arange(data["entropy"].shape[1])

    fig, ax = plt.subplots(figsize=SINGLE)
    _plot_band(ax, rounds, data["entropy"], correct, label="Correct", color=BLUE)
    _plot_band(ax, rounds, data["entropy"], ~correct, label="Incorrect", color=ORANGE)
    ax.set_xlabel("Round")
    ax.set_ylabel("Entropy")
    ax.set_xlim(rounds[0], rounds[-1])
    ax.set_ylim(0, 1)
    clean_axes(ax)
    finish_figure(fig, ax, "Exploration entropy", legend=True, legend_ncol=2)
    _save(fig, out, "exploration_entropy_collapse.png")

    fig, ax = plt.subplots(figsize=SINGLE)
    _plot_band(ax, rounds, data["frac_B"], correct, label="Correct", color=BLUE)
    _plot_band(ax, rounds, data["frac_B"], ~correct, label="Incorrect", color=ORANGE)
    ax.set_xlabel("Round")
    ax.set_ylabel(r"Fraction testing $b$")
    ax.set_xlim(rounds[0], rounds[-1])
    ax.set_ylim(0, 1.1)
    clean_axes(ax)
    finish_figure(fig, ax, "Testing the better arm", legend=True, legend_ncol=2)
    _save(fig, out, "better_arm_exploration.png")

    fig, ax = plt.subplots(figsize=SINGLE)
    _plot_band(ax, rounds, data["oracle_value"], correct, label="Correct", color=BLUE)
    _plot_band(ax, rounds, data["oracle_value"], ~correct, label="Incorrect", color=ORANGE)
    ax.axhline(0.50, label=r"$p_s$", **reference_line_style(linestyle="--"))
    ax.axhline(0.55, label=r"$p_b$", **reference_line_style())
    ax.set_xlabel("Round")
    ax.set_ylabel("Oracle belief")
    ax.set_xlim(rounds[0], rounds[-1])
    ax.set_ylim(0.25, 0.75)
    clean_axes(ax)
    finish_figure(fig, ax, "Oracle belief", legend=True, legend_ncol=4)
    _save(fig, out, "adaptive_oracle_lock_in.png")
