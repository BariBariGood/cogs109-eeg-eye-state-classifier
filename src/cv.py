"""Cross-validation fold-index generators.

Three schemes are provided so the modeling notebook can compare them honestly:

* ``blocked_kfold_indices`` — contiguous time blocks. Each fold's test set is
  one ~equal-sized chunk of the time-ordered training partition; the train
  set is the remaining time. This respects the temporal autocorrelation in
  the EEG recording (per-channel lag-1 r ≈ 0.97; the binary label is even
  more persistent at lag-1 r ≈ 0.997). Downside: with only 5 macro-blocks
  on this single-subject recording, individual folds can land on segments
  with very different class balance from the overall training partition
  (folds 3/4 in particular swing to 20% / 66% class-0 vs the partition's
  ~46.5% class-0), which collapses the model's ability to generalise to
  those folds.
* ``shuffled_kfold_indices`` — i.i.d. random k-fold. This is the standard
  sklearn ``KFold(shuffle=True)``-style split. On a time series with strong
  autocorrelation this leaks neighbors into the test fold and inflates the
  estimated generalisation accuracy.
* ``stratified_blocked_kfold_indices`` — a middle ground. Cut the
  training partition into many short contiguous segments (default 50), then
  assign segments to folds so each fold's class balance is close to the
  overall balance. Temporal locality is still preserved at the segment
  granularity (each segment is ~238 samples ≈ 1.86 s contiguous), but
  fold-level class imbalance is rebalanced.
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


def stratified_blocked_kfold_indices(
    labels: np.ndarray,
    n_splits: int = 5,
    n_segments: int = 50,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """K-fold indices that preserve temporal locality but balance class per fold.

    The training partition is cut into ``n_segments`` contiguous segments of
    roughly equal length (any remainder samples are folded into the final
    segment). Each segment is labelled by its class-1 proportion, the
    segments are sorted by that proportion, and then assigned to folds with
    a deterministic snake/round-robin pattern so that each fold receives one
    segment from each ``class-1-proportion`` quintile (or n-tile) of the
    sorted list. Within each chunk of ``n_splits`` consecutive sorted
    segments the assignment to fold indices is a random permutation seeded
    by ``seed`` (so the algorithm is deterministic).

    The result is a set of folds where:

    * Test indices for each fold are the union of contiguous segments —
      locality is preserved at the segment granularity (~1.86 s per segment
      at 128 Hz with 50 segments on the n=11,917 train partition).
    * Each sample appears in exactly one test fold; train indices are the
      complement.
    * Each fold's class balance is close to the overall class balance,
      which avoids the failure mode of naive ``blocked_kfold_indices`` on
      this single-subject recording (folds 3/4 land on segments with 20% /
      66% class-0 vs ~54% overall and the classifier cannot generalise).

    Parameters
    ----------
    labels:
        1-D array of binary labels (``0`` / ``1``) over the training
        partition, in original sample order.
    n_splits:
        Number of CV folds. Must be ≥ 2. ``n_segments`` must be a multiple
        of ``n_splits`` so every fold gets the same number of segments.
    n_segments:
        Number of contiguous segments to cut the training partition into.
        Defaults to 50, which gives ~238 samples (~1.86 s at 128 Hz) per
        segment on the n=11,917 train partition.
    seed:
        Seed for the deterministic per-chunk fold-id permutation.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    labels = np.asarray(labels).astype(int).ravel()
    n_samples = labels.shape[0]
    if n_samples < n_segments:
        raise ValueError("n_samples must be >= n_segments")
    if n_segments < n_splits:
        raise ValueError("n_segments must be >= n_splits")
    if n_segments % n_splits != 0:
        raise ValueError(
            f"n_segments ({n_segments}) must be a multiple of n_splits ({n_splits})"
        )

    seg_size = n_samples // n_segments
    seg_starts = (np.arange(n_segments, dtype=np.int64) * seg_size)
    seg_ends = np.concatenate([seg_starts[1:], [np.int64(n_samples)]])

    seg_props = np.empty(n_segments, dtype=np.float64)
    for s in range(n_segments):
        seg_props[s] = float((labels[seg_starts[s]:seg_ends[s]] == 1).mean())

    # Stable sort by class-1 proportion so the same input always yields the
    # same ordering (important for determinism when several segments tie).
    sort_order = np.argsort(seg_props, kind="stable")

    rng = np.random.default_rng(seed)
    seg_to_fold = np.empty(n_segments, dtype=np.int64)
    n_chunks = n_segments // n_splits
    for chunk_i in range(n_chunks):
        perm = rng.permutation(n_splits).astype(np.int64)
        chunk_start = chunk_i * n_splits
        for k in range(n_splits):
            seg_to_fold[sort_order[chunk_start + k]] = perm[k]

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    all_indices = np.arange(n_samples, dtype=np.int64)
    for fold_id in range(n_splits):
        test_seg_ids = np.where(seg_to_fold == fold_id)[0]
        if len(test_seg_ids) == 0:
            raise RuntimeError(
                f"fold {fold_id} received no segments — check n_segments / n_splits"
            )
        test_parts = [
            np.arange(seg_starts[s], seg_ends[s], dtype=np.int64)
            for s in test_seg_ids
        ]
        test_idx = np.sort(np.concatenate(test_parts))
        train_idx = np.setdiff1d(all_indices, test_idx, assume_unique=True)
        folds.append((train_idx.astype(np.int64), test_idx.astype(np.int64)))
    return folds
