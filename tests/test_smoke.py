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


def _synthetic_block_labels(n: int = 12000, n_blocks: int = 24, seed: int = 0) -> np.ndarray:
    """Build a label sequence with strong block structure but a 50/50 overall mix.

    Mirrors the UCI EEG eye-state recording pattern: contiguous runs of one
    class followed by contiguous runs of the other.
    """
    rng = np.random.default_rng(seed)
    block_lens = rng.integers(low=200, high=900, size=n_blocks)
    block_lens = (block_lens * (n / block_lens.sum())).astype(int)
    block_lens[-1] += n - int(block_lens.sum())
    out = np.empty(n, dtype=int)
    pos = 0
    label = 0
    for L in block_lens:
        out[pos : pos + L] = label
        pos += L
        label = 1 - label
    return out


def test_stratified_blocked_folds_deterministic():
    labels = _synthetic_block_labels(n=11917, n_blocks=24, seed=0)
    a = cv_mod.stratified_blocked_kfold_indices(
        labels, n_splits=5, n_segments=50, seed=42
    )
    b = cv_mod.stratified_blocked_kfold_indices(
        labels, n_splits=5, n_segments=50, seed=42
    )
    assert len(a) == len(b) == 5
    for (tr1, te1), (tr2, te2) in zip(a, b):
        assert np.array_equal(tr1, tr2)
        assert np.array_equal(te1, te2)


def test_stratified_blocked_folds_balanced():
    labels = _synthetic_block_labels(n=11917, n_blocks=24, seed=0)
    overall_c0 = float((labels == 0).mean())
    folds = cv_mod.stratified_blocked_kfold_indices(
        labels, n_splits=5, n_segments=50, seed=42
    )
    for fold_i, (_, te) in enumerate(folds):
        fold_c0 = float((labels[te] == 0).mean())
        assert abs(fold_c0 - overall_c0) < 0.05, (
            f"fold {fold_i} class-0 pct {fold_c0:.3f} differs from overall "
            f"{overall_c0:.3f} by more than 5 percentage points"
        )


def test_stratified_blocked_folds_cover_all_samples_exactly_once():
    labels = _synthetic_block_labels(n=11917, n_blocks=24, seed=0)
    folds = cv_mod.stratified_blocked_kfold_indices(
        labels, n_splits=5, n_segments=50, seed=42
    )
    seen = np.concatenate([te for _, te in folds])
    assert sorted(seen.tolist()) == list(range(len(labels)))
    # And train / test are disjoint within each fold.
    for tr, te in folds:
        assert len(np.intersect1d(tr, te)) == 0
        assert len(tr) + len(te) == len(labels)


def test_stratified_blocked_folds_segment_count_validation():
    labels = _synthetic_block_labels(n=11917, n_blocks=24, seed=0)
    # n_segments must be a multiple of n_splits.
    with pytest.raises(ValueError):
        cv_mod.stratified_blocked_kfold_indices(
            labels, n_splits=5, n_segments=49, seed=42
        )


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
