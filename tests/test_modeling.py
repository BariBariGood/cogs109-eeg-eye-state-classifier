"""Phase-B smoke tests for the modeling and evaluation modules.

These run on the frozen Phase-A processed splits and exercise every entry
point in :mod:`src.models` / :mod:`src.evaluate`. Total pytest budget for
the modeling tests is well under 60 seconds.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data import CHANNELS, LABEL_COL  # noqa: E402
from src.evaluate import cv_score  # noqa: E402
from src.models import (  # noqa: E402
    PCRClassifier,
    fit_knn,
    fit_lda,
    fit_pca_lda,
    fit_pcr_classifier,
    score,
)

PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
TRAIN_CSV = os.path.join(PROCESSED_DIR, "eeg_train.csv")
TEST_CSV = os.path.join(PROCESSED_DIR, "eeg_test.csv")
CV_FOLDS = os.path.join(PROCESSED_DIR, "cv_folds.json")


def _load_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    X_train = train[list(CHANNELS)].to_numpy()
    y_train = train[LABEL_COL].to_numpy().astype(int)
    X_test = test[list(CHANNELS)].to_numpy()
    y_test = test[LABEL_COL].to_numpy().astype(int)
    return X_train, y_train, X_test, y_test


@pytest.mark.skipif(
    not os.path.exists(TRAIN_CSV),
    reason="Phase A processed splits not present (run scripts/preprocess.py)",
)
def test_lda_baseline_runs():
    X_train, y_train, X_test, y_test = _load_split()
    model = fit_lda(X_train, y_train)
    metrics = score(model, X_test, y_test)
    assert set(metrics.keys()) == {
        "accuracy",
        "sensitivity",
        "specificity",
        "confusion_matrix",
    }
    assert 0.0 <= metrics["accuracy"] <= 1.0
    preds = model.predict(X_test)
    assert preds.shape == y_test.shape
    assert set(np.unique(preds).tolist()).issubset({0, 1})


@pytest.mark.skipif(
    not os.path.exists(TRAIN_CSV),
    reason="Phase A processed splits not present (run scripts/preprocess.py)",
)
def test_knn_baseline_runs():
    X_train, y_train, X_test, y_test = _load_split()
    model = fit_knn(X_train, y_train, k=11)
    metrics = score(model, X_test, y_test)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    preds = model.predict(X_test)
    assert preds.shape == y_test.shape
    assert set(np.unique(preds).tolist()).issubset({0, 1})


@pytest.mark.skipif(
    not os.path.exists(TRAIN_CSV) or not os.path.exists(CV_FOLDS),
    reason="Phase A processed splits or CV folds not present",
)
def test_cv_score_deterministic():
    X_train, y_train, _, _ = _load_split()
    with open(CV_FOLDS) as f:
        folds_blob = json.load(f)
    folds = [
        (np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64))
        for tr, te in folds_blob["blocked"][:2]
    ]
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    def factory():
        return LinearDiscriminantAnalysis(solver="svd")

    r1 = cv_score(factory, X_train, y_train, folds)
    r2 = cv_score(factory, X_train, y_train, folds)
    assert r1["accuracy"] == r2["accuracy"]
    assert r1["sensitivity"] == r2["sensitivity"]
    assert r1["specificity"] == r2["specificity"]
    for f1, f2 in zip(r1["per_fold"], r2["per_fold"]):
        assert f1["accuracy"] == f2["accuracy"]
        assert f1["confusion_matrix"] == f2["confusion_matrix"]


def test_pcr_classifier_predict_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 14))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    model = fit_pcr_classifier(X, y, n_components=5)
    assert isinstance(model, PCRClassifier)
    preds = model.predict(X)
    assert preds.shape == (200,)
    assert set(np.unique(preds).tolist()).issubset({0, 1})
    proba = model.predict_proba(X)
    assert proba.shape == (200, 2)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


def test_pca_lda_pipeline_runs():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 14))
    y = (X[:, 0] > 0).astype(int)
    pipe = fit_pca_lda(X, y, n_components=5)
    preds = pipe.predict(X)
    assert preds.shape == (150,)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
