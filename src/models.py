"""Supervised model definitions for the EEG eye-state classifier.

The model palette is locked to the COGS 109 Spring 2026 study guide:

* ``fit_lda`` — Linear Discriminant Analysis (sklearn ``LinearDiscriminantAnalysis``).
* ``fit_knn`` — k-Nearest Neighbors classifier (sklearn ``KNeighborsClassifier``).
* ``fit_pca_lda`` — PCA-then-LDA pipeline (dimensionality reduction + LDA).
* ``fit_pcr_classifier`` — Principal Components Regression as a classifier.
  Regress 0/1 labels onto the top-``n_components`` PCA scores with ordinary
  least squares, threshold at 0.5. Wrapped to expose the standard
  ``fit / predict / predict_proba`` surface so it interoperates with the
  evaluation helpers in :mod:`src.evaluate`.

``score`` computes the same metric bundle for every model in the palette:
accuracy, sensitivity (TPR for closed=1), specificity (TNR for open=0), and
the 2x2 confusion matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


def fit_lda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    shrinkage: bool = False,
) -> LinearDiscriminantAnalysis:
    """Fit Linear Discriminant Analysis on z-scored channel features.

    Defaults to the closed-form SVD solver. Setting ``shrinkage=True``
    switches to the ``lsqr`` solver with automatic Ledoit-Wolf shrinkage,
    which is the variant covered alongside LDA in the study guide.
    """
    if shrinkage:
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    else:
        model = LinearDiscriminantAnalysis(solver="svd")
    model.fit(X_train, y_train)
    return model


def fit_knn(
    X_train: np.ndarray, y_train: np.ndarray, k: int
) -> KNeighborsClassifier:
    """Fit a uniform-weight k-Nearest Neighbors classifier with ``k`` neighbors."""
    model = KNeighborsClassifier(
        n_neighbors=k, weights="uniform", algorithm="auto"
    )
    model.fit(X_train, y_train)
    return model


def fit_pca_lda(
    X_train: np.ndarray, y_train: np.ndarray, n_components: int
) -> Pipeline:
    """Fit a PCA(n_components) -> LDA pipeline.

    Uses the default LDA SVD solver; PCA's ``random_state`` is fixed for
    deterministic component signs across runs.
    """
    pipe = Pipeline(
        steps=[
            ("pca", PCA(n_components=n_components, random_state=42)),
            ("lda", LinearDiscriminantAnalysis(solver="svd")),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


@dataclass
class PCRClassifier:
    """Principal Components Regression used as a binary classifier.

    Fits ``LinearRegression`` on the top-``n_components`` PCA scores predicting
    ``y`` encoded as 0/1, then thresholds the continuous prediction at 0.5 to
    recover a class label. The continuous prediction itself is exposed via
    ``predict_proba`` (clipped into ``[0, 1]``) so it interoperates with the
    same evaluation surface as the sklearn classifiers in this module.
    """

    n_components: int

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PCRClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.pca_ = PCA(n_components=self.n_components, random_state=42)
        scores = self.pca_.fit_transform(X)
        self.reg_ = LinearRegression()
        self.reg_.fit(scores, y)
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.pca_.transform(np.asarray(X, dtype=float))
        return self.reg_.predict(scores)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self._raw_predict(X) >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = np.clip(self._raw_predict(X), 0.0, 1.0)
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)


def fit_pcr_classifier(
    X_train: np.ndarray, y_train: np.ndarray, n_components: int
) -> PCRClassifier:
    """Fit :class:`PCRClassifier` (PCA -> linear regression -> 0.5 threshold)."""
    return PCRClassifier(n_components=n_components).fit(X_train, y_train)


def score(
    estimator: Any, X: np.ndarray, y: np.ndarray
) -> Dict[str, Any]:
    """Compute the standard metric bundle for any palette estimator.

    Returns a dict with ``accuracy``, ``sensitivity`` (TPR for closed=1),
    ``specificity`` (TNR for open=0), and ``confusion_matrix`` (2x2 list:
    rows = true, cols = predicted, order ``[0, 1]``).
    """
    y_true = np.asarray(y).ravel()
    y_pred = np.asarray(estimator.predict(X)).ravel()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    accuracy = float((tn + tp) / max(tn + fp + fn + tp, 1))
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": cm.tolist(),
    }
