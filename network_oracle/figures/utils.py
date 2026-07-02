"""Small plotting helpers shared across study figures."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from network_oracle.figures.style import GRAY


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


def _mean_ci(data: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
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
    """Smooth a rectangular field with piecewise-linear interpolation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)

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
    """Draw the same smooth heatmap style throughout the project."""
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
