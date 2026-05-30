"""Shared figure helpers so the EDA notebook stays uncluttered.

Conventions
-----------
* DPI 120 for raster output (poster-readable, but not huge).
* Tight layout, sans-serif font, light grid.
* All saves go under ``figures/`` at the repo root unless an absolute path is
  passed.
"""

from __future__ import annotations

import os
from typing import Iterable

import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(REPO_ROOT, "figures")
DPI = 120

_RC_APPLIED = False


def apply_style() -> None:
    """Apply a consistent matplotlib style (called once per notebook)."""
    global _RC_APPLIED
    if _RC_APPLIED:
        return
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
        }
    )
    _RC_APPLIED = True


def figure_path(name: str) -> str:
    """Resolve ``name`` (e.g. ``"01_class_balance.png"``) under ``figures/``."""
    os.makedirs(FIG_DIR, exist_ok=True)
    if os.path.isabs(name):
        return name
    return os.path.join(FIG_DIR, name)


def save_fig(fig: plt.Figure, name: str) -> str:
    """Save ``fig`` to ``figures/<name>`` and return the resolved path."""
    path = figure_path(name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    return path


def annotate_axes(
    ax: plt.Axes,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> plt.Axes:
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return ax


def channel_grid(channels: Iterable[str], rows: int = 2, cols: int = 7):
    """Convenience: 2x7 subplot grid for the 14 channels."""
    apply_style()
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.2),
                             sharex=False, sharey=False)
    return fig, axes.flatten()
