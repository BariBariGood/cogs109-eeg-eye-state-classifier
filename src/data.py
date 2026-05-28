"""Data loading, cleaning, splitting, and scaling for the EEG eye-state project.

All routines here are deterministic and pure. The numbers in the docstrings
(14,980 raw rows, 14,976 after cleaning, etc.) refer to UCI ML Repository #264
and are asserted in tests and in the fetch notebook.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd

# Canonical channel order from the UCI dataset header.
CHANNELS: tuple[str, ...] = (
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
    "P8", "T8", "FC6", "F4", "F8", "AF4",
)
LABEL_COL: str = "eyeDetection"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_CSV_DEFAULT = os.path.join(REPO_ROOT, "data", "raw", "eeg_eye_state.csv")

logger = logging.getLogger(__name__)


def fetch_raw(csv_path: str | None = None) -> pd.DataFrame:
    """Load the raw EEG eye-state dataset.

    Tries to read the canonical raw CSV from disk first (idempotent / offline
    reproducibility). If that file does not exist, falls back to
    ``ucimlrepo.fetch_ucirepo(id=264)``. Returns a DataFrame with 14 channel
    columns followed by the ``eyeDetection`` label, in canonical order.
    """
    path = csv_path or RAW_CSV_DEFAULT
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        from ucimlrepo import fetch_ucirepo  # imported lazily

        repo = fetch_ucirepo(id=264)
        features = repo.data.features.copy()
        targets = repo.data.targets.copy()
        df = pd.concat([features, targets], axis=1)

    expected_cols = list(CHANNELS) + [LABEL_COL]
    df = df[expected_cols].reset_index(drop=True)
    return df


def clean(df: pd.DataFrame, z_threshold: float = 4.0) -> pd.DataFrame:
    """Drop rows where any channel has global |z| > ``z_threshold``.

    Z-scores are computed across all rows of the input (population stats), and
    a row is dropped if ANY channel exceeds the threshold. On UCI #264 with
    the default threshold of 4.0 this drops exactly 4 rows, returning 14,976.
    """
    channel_data = df[list(CHANNELS)].astype(float)
    mu = channel_data.mean(axis=0)
    sd = channel_data.std(axis=0, ddof=0)
    z = (channel_data - mu) / sd
    keep_mask = (z.abs() <= z_threshold).all(axis=1)
    n_drop = int((~keep_mask).sum())
    logger.info("clean(): dropping %d rows on |z|>%.1f", n_drop, z_threshold)
    return df.loc[keep_mask].reset_index(drop=True)


def split_chronological(
    df: pd.DataFrame,
    test_frac: float = 0.20,
    seam_gap: int = 64,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological 80/20 split with an optional sample-gap at the seam.

    The seam gap defuses lag-1 leakage between the last train sample and the
    first test sample. With the default ``seam_gap=64`` on 14,976 cleaned rows,
    train has ~11,917 rows and test has ~2,995 rows.
    """
    n = len(df)
    test_size = int(round(n * test_frac))
    test_start = max(0, n - test_size)
    train_end = max(0, test_start - seam_gap)
    train = df.iloc[:train_end].reset_index(drop=True)
    test = df.iloc[test_start:].reset_index(drop=True)
    return train, test


def split_shuffled(
    df: pd.DataFrame,
    test_frac: float = 0.20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Random 80/20 split. This is the leakage-prone scheme — used for contrast."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_test = int(round(len(df) * test_frac))
    test_idx = np.sort(idx[:n_test])
    train_idx = np.sort(idx[n_test:])
    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    return train, test


def fit_scaler(train_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Fit per-channel z-score parameters on the training rows.

    Returns a JSON-serialisable mapping ``{channel: {"mu": float, "sd": float}}``.
    """
    scaler: dict[str, dict[str, float]] = {}
    for ch in CHANNELS:
        col = train_df[ch].astype(float)
        sd = float(col.std(ddof=0))
        if sd == 0.0:
            sd = 1.0
        scaler[ch] = {"mu": float(col.mean()), "sd": sd}
    return scaler


def apply_scaler(
    df: pd.DataFrame, scaler: dict[str, dict[str, float]]
) -> pd.DataFrame:
    """Apply a fitted per-channel z-score scaler. The label column is passed through."""
    out = df.copy()
    for ch, params in scaler.items():
        out[ch] = (df[ch].astype(float) - params["mu"]) / params["sd"]
    return out


def save_scaler(scaler: dict[str, dict[str, float]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(scaler, f, indent=2, sort_keys=True)


def load_scaler(path: str) -> dict[str, dict[str, float]]:
    with open(path) as f:
        return json.load(f)
