#!/usr/bin/env python3
"""Regenerate ``figures/11_knn_k_sweep.png`` with all three CV schemes.

The reworked project narrative treats KNN model selection (sweeping ``k`` to
minimise 5-fold CV error) as the central story, and compares the same
model-selection procedure under three different CV schemes:

* shuffled 5-fold CV — leaky baseline (orange)
* naive blocked 5-fold CV — leakage-resistant but per-fold class-imbalanced
  on this single-subject recording (blue)
* stratified blocked 5-fold CV — leakage-resistant AND class-balanced; the
  honest evaluation (green)

The figure plots mean accuracy ± std-dev vs ``k`` (log-spaced grid) for
each scheme on the same axes, and annotates each scheme's argmax with a
marker + the picked accuracy. Colors are chosen to match the palette used
elsewhere in the modelling notebook (figures 11–15) so the visual identity
stays consistent across the report and poster.

The script is reproducible: it loads the cached training partition, the
cached fold-index sets from ``data/processed/cv_folds.json``, and the same
KNN factory used in ``notebooks/02_modeling.ipynb``. Re-running it will
overwrite ``figures/11_knn_k_sweep.png`` deterministically given a fixed
seed.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data import CHANNELS, LABEL_COL
from src.evaluate import cv_score
from src.plotting import apply_style, save_fig

from sklearn.neighbors import KNeighborsClassifier


PROC_DIR = os.path.join(REPO_ROOT, "data", "processed")
FIG_DIR = os.path.join(REPO_ROOT, "figures")
TABLE_DIR = os.path.join(REPO_ROOT, "tables")

# Log-spaced k grid for model selection. Matches the grid described in the
# methodology section of the report; small odd k captures the leakage-driven
# autocorrelation effect, large k damps it out.
K_GRID = [1, 3, 5, 7, 11, 15, 21, 31, 51, 75, 99, 151, 201]

# Colour palette — kept in sync with figure 14 (three-way CV comparison) so
# the same scheme always uses the same colour.
COLOR_SHUFFLED = "#dd8452"          # warm orange (leaky baseline)
COLOR_NAIVE_BLOCKED = "#4c72b0"     # blue (naive blocked)
COLOR_STRATIFIED_BLOCKED = "#55a868"  # green (stratified blocked — honest)

MAJORITY_BASELINE = 0.5512


def _knn_factory(k: int):
    """Return a fresh KNN estimator with ``n_neighbors=k`` on each call."""
    return lambda: KNeighborsClassifier(n_neighbors=k, weights="uniform")


def _load_train():
    train = pd.read_csv(os.path.join(PROC_DIR, "eeg_train.csv"))
    X = train[list(CHANNELS)].to_numpy()
    y = train[LABEL_COL].to_numpy().astype(int)
    return X, y


def _load_folds():
    with open(os.path.join(PROC_DIR, "cv_folds.json")) as f:
        blob = json.load(f)
    def _as_folds(name):
        return [
            (np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64))
            for tr, te in blob[name]
        ]
    return {
        "shuffled": _as_folds("shuffled"),
        "naive_blocked": _as_folds("blocked"),
        "stratified_blocked": _as_folds("stratified_blocked"),
    }


def _sweep(X, y, folds, label):
    rows = []
    for k in K_GRID:
        result = cv_score(_knn_factory(k), X, y, folds)
        acc = result["accuracy"]
        rows.append({
            "scheme": label,
            "k": k,
            "mean": acc["mean"],
            "std": acc["std"],
        })
        print(f"  {label:>22s}  k={k:>3d}  acc={acc['mean']:.4f} ± {acc['std']:.4f}")
    return pd.DataFrame(rows)


def main():
    apply_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    X, y = _load_train()
    folds = _load_folds()

    print("running KNN k-sweep under three CV schemes ...")
    df_shuf = _sweep(X, y, folds["shuffled"], "shuffled")
    df_naive = _sweep(X, y, folds["naive_blocked"], "naive blocked")
    df_strat = _sweep(X, y, folds["stratified_blocked"], "stratified blocked")

    # Persist a tidy CSV of the swept numbers so the figure is reproducible
    # and reviewable as a table too.
    sweep_df = pd.concat([df_shuf, df_naive, df_strat], ignore_index=True)
    sweep_csv = os.path.join(TABLE_DIR, "04_knn_k_sweep.csv")
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"wrote {sweep_csv}")

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    def _plot(d, color, marker, label):
        ax.errorbar(
            d["k"], d["mean"], yerr=d["std"],
            marker=marker, color=color, capsize=3, linewidth=1.6,
            markersize=6, label=label,
        )
        # Annotate the argmax for this scheme.
        i = int(d["mean"].idxmax())
        kx = int(d.loc[i, "k"])
        my = float(d.loc[i, "mean"])
        sy = float(d.loc[i, "std"])
        ax.scatter([kx], [my], s=90, facecolor="white", edgecolor=color,
                   linewidth=1.8, zorder=5)
        ax.annotate(
            f"k={kx}\n{my:.3f} ± {sy:.3f}",
            xy=(kx, my),
            xytext=(8, 8), textcoords="offset points",
            fontsize=8, color=color, fontweight="bold",
        )

    _plot(df_shuf, COLOR_SHUFFLED, "s", "shuffled 5-fold CV (leaky)")
    _plot(df_naive, COLOR_NAIVE_BLOCKED, "o", "naive blocked 5-fold CV")
    _plot(df_strat, COLOR_STRATIFIED_BLOCKED, "D",
          "stratified blocked 5-fold CV (honest)")

    ax.axhline(MAJORITY_BASELINE, color="gray", linestyle="--",
               linewidth=0.8, label=f"majority-class baseline ({MAJORITY_BASELINE:.4f})")

    ax.set_xscale("log")
    ax.set_xlabel("k (number of neighbours, log scale)")
    ax.set_ylabel("5-fold CV accuracy")
    ax.set_title("KNN model selection under three CV schemes")
    ax.set_ylim(0.30, 1.02)
    ax.legend(loc="lower left", fontsize=9)

    out = save_fig(fig, "11_knn_k_sweep.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
