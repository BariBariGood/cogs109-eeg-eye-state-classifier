"""Cross-validation and holdout evaluation orchestration.

The CV routine takes a list of precomputed ``(train_idx, test_idx)`` folds —
typically the frozen blocked or shuffled folds loaded from
``data/processed/cv_folds.json`` — so the modeling notebook never has to
re-derive the splits. Every estimator built by :mod:`src.models` exposes the
standard ``fit`` / ``predict`` surface and is wrapped here through a
``model_factory`` callable that returns a fresh, untrained estimator on each
call.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np

from .models import score


FoldList = Iterable[Tuple[np.ndarray, np.ndarray]]


def cv_score(
    model_factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    folds: FoldList,
) -> Dict[str, Any]:
    """Run k-fold cross-validation with precomputed indices.

    ``model_factory()`` must return a fresh, unfitted estimator on each call.
    For each fold we fit on ``X[train_idx]`` / ``y[train_idx]``, evaluate on
    ``X[test_idx]`` / ``y[test_idx]``, and aggregate the per-fold metric
    bundle returned by :func:`src.models.score`. The result is deterministic
    given a deterministic ``model_factory`` and ``folds``.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    per_fold: List[Dict[str, Any]] = []
    accs: List[float] = []
    sens: List[float] = []
    spec: List[float] = []
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        metrics = score(model, X[test_idx], y[test_idx])
        per_fold.append({"fold": fold_i, **metrics})
        accs.append(metrics["accuracy"])
        sens.append(metrics["sensitivity"])
        spec.append(metrics["specificity"])

    def _stats(vals: List[float]) -> Dict[str, float]:
        arr = np.asarray(vals, dtype=float)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}

    return {
        "accuracy": _stats(accs),
        "sensitivity": _stats(sens),
        "specificity": _stats(spec),
        "per_fold": per_fold,
        "n_folds": len(per_fold),
    }


def final_holdout_score(
    estimator: Any, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, Any]:
    """Evaluate a fitted estimator on the chronological holdout test set."""
    return score(estimator, X_test, y_test)
