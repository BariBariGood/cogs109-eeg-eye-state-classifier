#!/usr/bin/env python3
"""End-to-end preprocessing CLI for the EEG eye-state project.

Reads ``data/raw/eeg_eye_state.csv`` (produced by
``notebooks/00_fetch_data.ipynb``), runs the full clean → split → scale →
fold-index pipeline, and writes deterministic artifacts to ``data/processed/``.

All files under ``data/processed/`` are byte-identical across runs (the CSVs,
the scaler JSONs, and ``cv_folds.json`` only depend on the raw CSV and the
fixed seed). The only field that changes between runs is the
``last_preprocessed_at`` timestamp inside ``data/raw/manifest.json``; the
``sha256`` and dataset dimensions in that manifest are preserved.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import cv as cv_mod  # noqa: E402
from src import data as data_mod  # noqa: E402

PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
SEED = 42
N_SPLITS = 5
# 100 contiguous segments → ~119 samples (~0.93 s) each at 128 Hz. We
# deliberately use a finer-grained M than the conservative default of 50:
# every fold then receives 20 segments evenly distributed across the
# class-1-proportion sort order, which keeps per-fold class balance within
# ~3 percentage points of the overall ~54% class-0 balance and lets the
# stratified scheme recover its expected ~15–25pp of accuracy that the
# 5-macro-block naive scheme artificially destroyed.
N_SEGMENTS_STRAT = 100
TEST_FRAC = 0.20
SEAM_GAP = 64

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("preprocess")


def _write_csv(df, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _folds_to_json(folds):
    return [[list(map(int, tr.tolist())), list(map(int, te.tolist()))] for tr, te in folds]


def main() -> int:
    log.info("loading raw data")
    raw = data_mod.fetch_raw()
    assert raw.shape == (14980, 15), f"unexpected raw shape: {raw.shape}"

    log.info("cleaning (drop |z|>4)")
    cleaned = data_mod.clean(raw)
    log.info("cleaned shape: %s", cleaned.shape)

    _write_csv(cleaned, os.path.join(PROCESSED_DIR, "eeg_clean.csv"))

    # Split A — chronological with seam gap (the "honest" split)
    log.info("computing Split A (chronological 80/20, seam gap=%d)", SEAM_GAP)
    train_a, test_a = data_mod.split_chronological(
        cleaned, test_frac=TEST_FRAC, seam_gap=SEAM_GAP
    )
    log.info("Split A: train=%d, test=%d", len(train_a), len(test_a))

    scaler_a = data_mod.fit_scaler(train_a)
    train_a_z = data_mod.apply_scaler(train_a, scaler_a)
    test_a_z = data_mod.apply_scaler(test_a, scaler_a)
    _write_csv(train_a_z, os.path.join(PROCESSED_DIR, "eeg_train.csv"))
    _write_csv(test_a_z, os.path.join(PROCESSED_DIR, "eeg_test.csv"))
    data_mod.save_scaler(scaler_a, os.path.join(PROCESSED_DIR, "scaler.json"))

    # Split B — shuffled (the "leakage-prone" split, for comparison only)
    log.info("computing Split B (shuffled 80/20, seed=%d)", SEED)
    train_b, test_b = data_mod.split_shuffled(
        cleaned, test_frac=TEST_FRAC, seed=SEED
    )
    log.info("Split B: train=%d, test=%d", len(train_b), len(test_b))

    scaler_b = data_mod.fit_scaler(train_b)
    train_b_z = data_mod.apply_scaler(train_b, scaler_b)
    test_b_z = data_mod.apply_scaler(test_b, scaler_b)
    _write_csv(train_b_z, os.path.join(PROCESSED_DIR, "eeg_train_shuffled.csv"))
    _write_csv(test_b_z, os.path.join(PROCESSED_DIR, "eeg_test_shuffled.csv"))
    data_mod.save_scaler(
        scaler_b, os.path.join(PROCESSED_DIR, "scaler_shuffled.json")
    )

    # CV folds over Split A's train partition (all three schemes, frozen)
    log.info(
        "computing CV fold indices (blocked + shuffled + stratified_blocked, k=%d)",
        N_SPLITS,
    )
    n_train_a = len(train_a)
    blocked_folds = cv_mod.blocked_kfold_indices(n_train_a, n_splits=N_SPLITS)
    shuffled_folds = cv_mod.shuffled_kfold_indices(
        n_train_a, n_splits=N_SPLITS, seed=SEED
    )
    y_train_a = train_a_z[data_mod.LABEL_COL].to_numpy().astype(int)
    stratified_blocked_folds = cv_mod.stratified_blocked_kfold_indices(
        y_train_a,
        n_splits=N_SPLITS,
        n_segments=N_SEGMENTS_STRAT,
        seed=SEED,
    )
    folds_payload = {
        "n_samples": int(n_train_a),
        "n_splits": int(N_SPLITS),
        "seed": int(SEED),
        "n_segments_stratified": int(N_SEGMENTS_STRAT),
        "blocked": _folds_to_json(blocked_folds),
        "shuffled": _folds_to_json(shuffled_folds),
        "stratified_blocked": _folds_to_json(stratified_blocked_folds),
    }
    with open(os.path.join(PROCESSED_DIR, "cv_folds.json"), "w") as f:
        json.dump(folds_payload, f)

    # Update the raw manifest's processed-at timestamp without touching the SHA.
    # We only write the manifest when its non-timestamp content would change;
    # otherwise we leave the file alone to avoid spurious diffs on re-runs.
    manifest_path = os.path.join(RAW_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {
            "source_url": "https://archive.ics.uci.edu/static/public/264/eeg+eye+state.zip",
            "n_rows": 14980,
            "n_cols": 15,
        }
    new_manifest = dict(manifest)
    new_manifest["last_preprocessed_at"] = datetime.now(timezone.utc).isoformat()
    # If the only thing that would change is the timestamp, keep the file as-is.
    cmp_old = {k: v for k, v in manifest.items() if k != "last_preprocessed_at"}
    cmp_new = {k: v for k, v in new_manifest.items() if k != "last_preprocessed_at"}
    if cmp_old != cmp_new or "last_preprocessed_at" not in manifest:
        with open(manifest_path, "w") as f:
            json.dump(new_manifest, f, indent=2, sort_keys=True)

    log.info("preprocess complete; artifacts under %s", PROCESSED_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
