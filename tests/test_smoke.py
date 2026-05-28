"""Phase-A smoke tests.

These run in a few seconds and are meant to catch regressions in the data
loading + CV scaffolding before the modeling work lands in Phase B.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import cv as cv_mod  # noqa: E402
from src import data as data_mod  # noqa: E402


def test_data_loads():
    df = data_mod.fetch_raw()
    assert df.shape == (14980, 15)
    assert list(df.columns) == list(data_mod.CHANNELS) + [data_mod.LABEL_COL]
    counts = df[data_mod.LABEL_COL].value_counts().to_dict()
    assert counts == {0: 8257, 1: 6723}
    assert df.isna().sum().sum() == 0


def test_clean_drops_outliers():
    df = data_mod.fetch_raw()
    cleaned = data_mod.clean(df)
    assert cleaned.shape == (14976, 15)
    # Sanity: no row in cleaned should exceed |z|>4 against the ORIGINAL stats.
    mu = df[list(data_mod.CHANNELS)].mean(axis=0)
    sd = df[list(data_mod.CHANNELS)].std(axis=0, ddof=0)
    z = (cleaned[list(data_mod.CHANNELS)] - mu) / sd
    assert (z.abs() <= 4.0).all().all()


def test_cv_folds_are_deterministic():
    n = 11917  # close to actual train-partition length
    a1 = cv_mod.blocked_kfold_indices(n, n_splits=5)
    a2 = cv_mod.blocked_kfold_indices(n, n_splits=5)
    for (tr1, te1), (tr2, te2) in zip(a1, a2):
        assert np.array_equal(tr1, tr2)
        assert np.array_equal(te1, te2)

    b1 = cv_mod.shuffled_kfold_indices(n, n_splits=5, seed=42)
    b2 = cv_mod.shuffled_kfold_indices(n, n_splits=5, seed=42)
    for (tr1, te1), (tr2, te2) in zip(b1, b2):
        assert np.array_equal(tr1, tr2)
        assert np.array_equal(te1, te2)


def test_cv_folds_cover_all_samples_exactly_once():
    n = 1000
    for folds in (
        cv_mod.blocked_kfold_indices(n, n_splits=5),
        cv_mod.shuffled_kfold_indices(n, n_splits=5, seed=42),
    ):
        seen = np.concatenate([te for _, te in folds])
        assert sorted(seen.tolist()) == list(range(n))


def test_blocked_folds_are_contiguous():
    folds = cv_mod.blocked_kfold_indices(1000, n_splits=5)
    for _, te in folds:
        assert np.array_equal(te, np.arange(te[0], te[-1] + 1))


def test_splits_disjoint():
    df = data_mod.fetch_raw()
    cleaned = data_mod.clean(df)
    train, test = data_mod.split_chronological(cleaned, test_frac=0.2, seam_gap=64)
    assert len(train) + len(test) <= len(cleaned)
    assert len(train) > 0 and len(test) > 0


def test_save_scaler_handles_bare_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scaler = {"AF3": {"mu": 0.0, "sd": 1.0}}
    data_mod.save_scaler(scaler, "scaler.json")
    assert (tmp_path / "scaler.json").exists()
    assert data_mod.load_scaler(str(tmp_path / "scaler.json")) == scaler


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
