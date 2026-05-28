# EEG Eye-State Classifier

A COGS 109 (Spring 2026) course project that classifies whether a subject's
eyes are open or closed from a 14-channel EEG recording.

## Project description

We use the public **EEG Eye State** dataset (UCI ML Repository #264) — a
single-subject, ~117-second continuous recording sampled at ~128 Hz, with each
sample labeled `0` (eyes open) or `1` (eyes closed). The dataset is 14,980 ×
15 (14 channel voltages + 1 binary label). Our work-plan walks the dataset
through cleaning, exploratory analysis, and supervised classification using
the methods taught in the course.

Source data: <https://archive.ics.uci.edu/dataset/264/eeg+eye+state>

## Status

**Phase A complete** — data + EDA + preprocessing.
**Phase B complete** — LDA / KNN / PCA→LDA / PCR-as-classifier evaluated
under three 5-fold cross-validation schemes in `notebooks/02_modeling.ipynb`:
shuffled (leaky), naive blocked (leakage-resistant but per-fold class-balance
pathology drops every model at or below the majority-class baseline), and
**stratified blocked** (leakage-resistant *and* class-balanced per fold —
the honest estimate). Stratified blocking recovers ~12–28 pp of accuracy
the naive scheme artificially destroyed; the residual stratified-vs-shuffled
gap is the true leakage signal (largest for KNN at ~20 pp). The headline
three-way comparison lives at `figures/14_cv_comparison_three_way.png` and
`tables/03_cv_accuracy_comparison.csv`.

## Repository layout

```
├── data/
│   ├── raw/                  # raw CSV pulled from UCI + manifest
│   └── processed/            # outlier-cleaned, z-scored splits + CV folds
├── figures/                  # EDA PNGs (committed)
├── tables/                   # EDA summary tables (committed)
├── notebooks/
│   ├── 00_fetch_data.ipynb   # downloads UCI #264, asserts shape, writes manifest
│   ├── 01_eda.ipynb          # exploratory analysis (11 figures, 2 tables)
│   └── 02_modeling.ipynb     # supervised models + blocked vs shuffled CV
├── scripts/
│   ├── preprocess.py         # full preprocessing CLI (idempotent)
│   └── _build_notebooks.py   # authoring helper for the .ipynb files
├── src/
│   ├── data.py               # loaders, cleaner, splitters, scaler
│   ├── cv.py                 # blocked + shuffled k-fold index generators
│   ├── models.py             # LDA / KNN / PCA→LDA / PCR-as-classifier
│   ├── evaluate.py           # CV orchestration + holdout scoring
│   └── plotting.py           # shared figure helpers
├── tests/
│   ├── test_smoke.py         # data + CV smoke tests
│   └── test_modeling.py      # model + evaluation smoke tests
├── Makefile
├── requirements.txt
└── LICENSE
```

## Reproduction

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute notebooks/00_fetch_data.ipynb --inplace
python scripts/preprocess.py
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/02_modeling.ipynb --inplace
pytest tests/
```

Or use the Makefile:

```bash
make fetch     # download the raw CSV via the fetch notebook
make data      # run scripts/preprocess.py (requires the raw CSV)
make figures   # execute the EDA notebook (depends on `data`)
make modeling  # execute the modeling notebook (depends on `data`)
make test      # run pytest
```

`make all` runs `data` + `figures` + `modeling`. `make fetch` is
intentionally a separate target because the raw CSV is already committed
under `data/raw/` — most contributors won't need to re-download it.

## Why three cross-validation schemes?

Adjacent samples in this recording have a lag-1 label autocorrelation of
**r ≈ 0.997** (see `figures/08_label_autocorrelation.png`), so shuffled
k-fold CV leaks neighboring samples into the test fold and inflates
accuracy. A naive 5-block time split fixes that leakage but, on this
single-subject recording, the five macro-blocks happen to span very
different class balances (folds 3 and 4 land on segments that are ~20%
and ~66% class-0 vs ~54% overall), which drives every model's naive
blocked accuracy at or below the majority-class baseline. Stratified
blocked CV cuts the train partition into 100 short contiguous segments
(~0.93 s each at 128 Hz) and redistributes them across folds so that
each fold's class balance is close to the overall balance, while keeping
test samples in contiguous bursts. All three schemes are frozen at
preprocessing time so the modeling notebook can report all three numbers
side by side.

## Authors

Ivan Del Rio (@BariBariGood) and Arkon Damadugula — UC San Diego, COGS 109,
Spring 2026.
