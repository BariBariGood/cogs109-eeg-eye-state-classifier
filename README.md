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

**Phase A complete — data + EDA + preprocessing.** Modeling
(LDA / KNN / PCA→LDA / PCR-as-classifier under both shuffled and blocked
k-fold cross-validation) is staged for `notebooks/02_modeling.ipynb` and is
not yet included in this checkpoint.

## Repository layout

```
├── data/
│   ├── raw/                  # raw CSV pulled from UCI + manifest
│   └── processed/            # outlier-cleaned, z-scored splits + CV folds
├── figures/                  # EDA PNGs (committed)
├── tables/                   # EDA summary tables (committed)
├── notebooks/
│   ├── 00_fetch_data.ipynb   # downloads UCI #264, asserts shape, writes manifest
│   └── 01_eda.ipynb          # exploratory analysis (11 figures, 2 tables)
├── scripts/
│   └── preprocess.py         # full preprocessing CLI (idempotent)
├── src/
│   ├── data.py               # loaders, cleaner, splitters, scaler
│   ├── cv.py                 # blocked + shuffled k-fold index generators
│   └── plotting.py           # shared figure helpers
├── tests/
│   └── test_smoke.py         # pytest smoke tests
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
pytest tests/
```

Or use the Makefile:

```bash
make fetch     # download the raw CSV via the fetch notebook
make data      # run scripts/preprocess.py (requires the raw CSV)
make figures   # execute the EDA notebook (depends on `data`)
make test      # run pytest
```

`make all` runs `data` + `figures`. `make fetch` is intentionally a separate
target because the raw CSV is already committed under `data/raw/` — most
contributors won't need to re-download it.

## Why two cross-validation schemes?

Adjacent samples in this recording have a lag-1 label autocorrelation of
**r ≈ 0.997** (see `figures/08_label_autocorrelation.png`). Shuffled k-fold
CV would leak neighboring samples into the test fold and inflate accuracy.
We therefore freeze both a shuffled and a time-blocked CV scheme during
preprocessing so the modeling notebook can report both numbers side by side.

## Authors

Ivan Del Rio (@BariBariGood) and Arkon Damadugula — UC San Diego, COGS 109,
Spring 2026.
