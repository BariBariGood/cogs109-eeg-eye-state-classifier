"""Cross-validation fold-index generators.

Two schemes are provided so the modeling notebook can compare them honestly:

* ``blocked_kfold_indices`` — contiguous time blocks. Each fold's test set is
  one ~equal-sized chunk of the time-ordered training partition; the train
  set is the remaining time. This respects the temporal autocorrelation in
  the EEG recording (lag-1 r ≈ 0.997).
* ``shuffled_kfold_indices`` — i.i.d. random k-fold. This is the standard
  sklearn ``KFold(shuffle=True)``-style split. On a time series with strong
  autocorrelation this leaks neighbors into the test fold and inflates the
  estimated generalisation accuracy.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def blocked_kfold_indices(
    n_samples: int, n_splits: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Contiguous time-block k-fold over ``range(n_samples)``.

    Returns a list of (train_idx, test_idx) pairs as ``np.ndarray`` of int64.
    Fold sizes are as equal as possible; the last fold absorbs the remainder.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n_samples < n_splits:
        raise ValueError("n_samples must be >= n_splits")

    indices = np.arange(n_samples, dtype=np.int64)
    fold_size = n_samples // n_splits
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        start = k * fold_size
        end = (k + 1) * fold_size if k < n_splits - 1 else n_samples
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_idx.astype(np.int64), test_idx.astype(np.int64)))
    return folds


def shuffled_kfold_indices(
    n_samples: int, n_splits: int = 5, seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Random-shuffled k-fold over ``range(n_samples)``.

    Deterministic given ``seed``. Each sample appears in exactly one test fold.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n_samples < n_splits:
        raise ValueError("n_samples must be >= n_splits")

    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples, dtype=np.int64)
    rng.shuffle(indices)
    fold_size = n_samples // n_splits
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        start = k * fold_size
        end = (k + 1) * fold_size if k < n_splits - 1 else n_samples
        test_idx = np.sort(indices[start:end])
        train_idx = np.sort(
            np.concatenate([indices[:start], indices[end:]])
        )
        folds.append((train_idx.astype(np.int64), test_idx.astype(np.int64)))
    return folds
