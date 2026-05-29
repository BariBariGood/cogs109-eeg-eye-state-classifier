# EEG Eye-State Classifier — An Honest Cross-Validation Study

> COGS 109 (Spring 2026) final project — UC San Diego — Mukamel.
> Authors: **Ivan Del Rio** ([@BariBariGood](https://github.com/BariBariGood))
> and **Anish Kondamadugula** ([@arkondamadugula](https://github.com/arkondamadugula)).

A small, reproducible project that classifies whether a subject's eyes
are open or closed from a 14-channel EEG recording — using **only** the
methods covered in the COGS 109 SP26 syllabus — and uses the dataset to
demonstrate how badly shuffled cross-validation can mislead an
unsuspecting analyst on autocorrelated time-series data. The headline
result is that our best honest classifier achieves **77.8% ± 2.7%**
stratified blocked CV accuracy, compared to the **97%** that the same
classifier reports under a leaky shuffled-CV protocol.

The full paper-style writeup lives at [`report/final_report.md`](report/final_report.md);
the academic poster lives at [`poster/poster.pptx`](poster/poster.pptx) with a
[PNG preview](poster/poster_preview.png).

## Hypothesis & contribution

**Hypothesis.** A linear or nearest-neighbour classifier trained on raw
14-channel EEG voltages can predict per-sample eye state above the
majority-class baseline of 55.12%, but accuracy estimates obtained
under shuffled k-fold cross-validation will substantially exceed honest
out-of-time estimates because of temporal leakage between adjacent
samples.

**Contribution.** The contribution of this project is *methodological*
rather than predictive. We provide a worked, fully reproducible
comparison between three k=5 cross-validation schemes — shuffled, naive
blocked, and stratified blocked — on the UCI #264 EEG Eye State dataset,
using only the four-model COGS 109 palette (LDA, KNN, PCA→LDA, PCR as a
binary classifier). We publish both the leaky and the honest accuracy
numbers side by side so future students of this dataset can calibrate
their expectations against a reference.

## Dataset

UCI Machine Learning Repository, dataset
[#264 — EEG Eye State](https://archive.ics.uci.edu/dataset/264/eeg+eye+state)
(Roesler, 2013). A single uninterrupted recording from one subject
wearing an Emotiv EPOC consumer-grade EEG headset.

- **Samples:** 14,980, sampled at ~128 Hz over ~117 seconds.
- **Channels:** 14 Emotiv electrodes — AF3, F7, F3, FC5, T7, P7, O1, O2,
  P8, T8, FC6, F4, F8, AF4.
- **Label:** binary `eyeDetection` — 0 = eyes open, 1 = eyes closed.
- **Class balance:** 55.12% open / 44.88% closed → majority baseline
  0.5512.
- **Label structure:** only **24 contiguous label runs** in the whole
  recording (i.e. 23 state transitions in ~117 s), with substantial
  variation in run length.
- **Autocorrelation:** lag-1 channel correlation ≈ 0.997; label
  autocorrelation remains > 0.8 out to ~100 samples (~0.8 s).

That last point is the structural reason shuffled CV leaks on this
dataset: the nearest sample in time is almost always in the same class,
so any classifier that exploits sample-to-sample similarity (e.g. KNN
with k=1) trivially recovers it from a shuffled training fold.

## Methodology (short version)

The full methodology lives in
[`report/final_report.md`](report/final_report.md). In short:

1. **Preprocessing.** Drop 4 outlier samples (any channel > 4 SD
   above mean), chronologically split 80/20 train/test with a
   64-sample seam gap, z-score channels using train-only statistics.
2. **Cross-validation schemes.** All k = 5:
   - **shuffled** — uniform-random split (leaky baseline);
   - **naive blocked** — 5 contiguous time chunks (leakage-resistant
     but class-imbalanced per fold);
   - **stratified blocked** — 100 short contiguous segments
     redistributed across folds to balance class proportion.
3. **Models** (COGS 109 palette only): LDA, KNN (k tuned), PCA → LDA
   (n tuned), PCR thresholded at 0.5 (n tuned). All hyperparameters
   selected by maximum mean **stratified blocked** CV accuracy.
4. **Evaluation.** For each model × scheme combination we report mean
   accuracy ± std-dev across folds; we also fit the four final models
   on the full training set and report confusion matrices on the
   chronological holdout.

No spectral / FFT features, no deep models, no off-syllabus methods.

## Headline results

| Model | Shuffled (leaky) | Naive blocked | **Stratified blocked (honest)** | Recovered | Residual leakage |
|---|---|---|---|---|---|
| LDA | 0.6471 ± 0.0063 | 0.4076 ± 0.1223 | **0.5911 ± 0.0474** | +18.4 pp | +5.6 pp |
| KNN (k=1) | 0.9728 ± 0.0037 | 0.5004 ± 0.1068 | **0.7778 ± 0.0274** | +27.7 pp | +19.5 pp |
| PCA→LDA (n=3) | 0.5574 ± 0.0147 | 0.4104 ± 0.0827 | **0.5339 ± 0.0438** | +12.4 pp | +2.4 pp |
| PCR (n=2) | 0.5528 ± 0.0087 | 0.4079 ± 0.1301 | **0.5250 ± 0.0176** | +11.7 pp | +2.8 pp |

> **Best honest model: KNN (k=1) at 77.8% ± 2.7%** stratified blocked CV.
> Same model under shuffled CV: 97.3% ± 0.4%. The 19.5 percentage-point
> gap is, we argue, mostly leakage.

Full numeric source: [`tables/03_cv_accuracy_comparison.csv`](tables/03_cv_accuracy_comparison.csv).

Headline figure:

![Three-way CV comparison](figures/14_cv_comparison_three_way.png)

The leakage gap is largest for KNN, which directly exploits
sample-to-sample similarity, and much smaller (2–6 pp) for LDA, PCA→LDA,
and PCR, which average over many samples before deciding.

## Repository layout

```
.
├── data/
│   ├── raw/                       # raw CSV pulled from UCI + manifest
│   └── processed/                 # outlier-cleaned, z-scored splits + CV folds
├── figures/                       # 15 committed PNG figures (EDA + modelling)
├── tables/                        # 3 committed CSV summary tables
├── notebooks/
│   ├── 00_fetch_data.ipynb        # downloads UCI #264, writes idempotent manifest
│   ├── 01_eda.ipynb               # exploratory analysis (11 figures, 2 tables)
│   └── 02_modeling.ipynb          # models + 3-way CV comparison + figures 11-15
├── poster/
│   ├── poster.pptx                # 48"×36" final poster (source of truth)
│   ├── poster.md                  # paste-ready Markdown version
│   ├── poster_preview.png         # rendered preview
│   └── README.md                  # poster-specific notes
├── report/
│   └── final_report.md            # paper-style writeup (~4000 words)
├── scripts/
│   ├── preprocess.py              # full preprocessing CLI (idempotent)
│   └── build_poster.py            # python-pptx builder for the poster (idempotent)
├── src/
│   ├── data.py                    # loaders, cleaner, splitters, scaler
│   ├── cv.py                      # blocked + shuffled + stratified-blocked k-fold
│   ├── models.py                  # LDA / KNN / PCA→LDA / PCR-as-classifier
│   ├── evaluate.py                # CV orchestration + holdout scoring
│   └── plotting.py                # shared figure helpers
├── tests/
│   ├── test_smoke.py              # data + CV smoke tests
│   ├── test_modeling.py           # model + evaluation smoke tests
│   └── test_poster.py             # poster builder produced-file + idempotency tests
├── Makefile
├── requirements.txt
└── LICENSE
```

## Quick start / reproducibility

From a clean clone:

```bash
git clone https://github.com/BariBariGood/cogs109-eeg-eye-state-classifier.git
cd cogs109-eeg-eye-state-classifier
pip install -r requirements.txt

make data        # fetch + preprocess (idempotent — won't re-download if up to date)
make modeling    # execute the modelling notebook (writes figures 11-15)
make poster      # regenerate poster/poster.pptx + poster/poster_preview.png
pytest tests/    # 18 tests, < 2 min
```

Or all-at-once: `make all` → `data` + `figures` + `modeling`, then
`make poster` for the .pptx.

Individual Makefile targets:

| Target          | Effect                                                    |
|-----------------|-----------------------------------------------------------|
| `make fetch`    | run `notebooks/00_fetch_data.ipynb` (downloads UCI #264)  |
| `make data`     | run `scripts/preprocess.py` (cleans + splits + scales)    |
| `make figures`  | run `notebooks/01_eda.ipynb` (writes figures 01–10)       |
| `make modeling` | run `notebooks/02_modeling.ipynb` (writes figures 11–15)  |
| `make poster`   | run `scripts/build_poster.py` (writes `poster/poster.*`)  |
| `make test`     | run pytest                                                |
| `make all`      | `data` + `figures` + `modeling`                           |
| `make clean`    | remove regeneratable outputs                              |

`scripts/build_poster.py` and `scripts/preprocess.py` are both
**idempotent** — running them twice in a row produces no diff. The
`.pptx` is re-packed with deterministic ZIP timestamps after generation,
and the preprocessing manifest is updated only when the SHA-256 of the
raw data has actually changed.

## Files / figures inventory

### Notebooks

| Notebook                          | What it does                                                                |
|-----------------------------------|-----------------------------------------------------------------------------|
| `notebooks/00_fetch_data.ipynb`   | downloads UCI #264 via `ucimlrepo`, writes raw CSV + SHA-256 manifest        |
| `notebooks/01_eda.ipynb`          | exploratory analysis — class balance, channel distributions, autocorrelation |
| `notebooks/02_modeling.ipynb`     | LDA / KNN / PCA→LDA / PCR models + 3-way CV comparison + holdout confusions  |

### Figures (`figures/`)

| Figure | Purpose |
|---|---|
| `01_class_balance.png`                  | dataset class balance (55.12% open / 44.88% closed) |
| `02_channel_boxplots.png`               | per-channel voltage boxplots (raw vs. cleaned) |
| `03_channel_histograms_raw.png`         | per-channel histograms before outlier removal |
| `03_channel_histograms_clean.png`       | per-channel histograms after outlier removal |
| `04_timeseries_with_label.png`          | a one-second EEG window with the eye-state label strip overlaid |
| `04b_timeseries_with_folds.png`         | per-fold sample assignments under all three CV schemes |
| `05_channel_correlation_heatmap.png`    | inter-channel Pearson correlation heatmap |
| `06_channel_dendrogram.png`             | hierarchical clustering of channels by correlation |
| `07_kmeans_vs_label.png`                | unsupervised K-means clustering vs. the eye-state label |
| `08_label_autocorrelation.png`          | label autocorrelation across lags — motivates blocked CV |
| `09_pca_scatter.png`                    | first two principal components, coloured by label |
| `10_lda_coefficients.png`               | LDA discriminant coefficients per channel |
| `11_knn_k_sweep.png`                    | KNN accuracy vs. k under stratified blocked CV |
| `12_pca_lda_components_sweep.png`       | PCA→LDA accuracy vs. number of components |
| `13_pcr_components_sweep.png`           | PCR accuracy vs. number of components |
| `14_cv_comparison_three_way.png`        | **headline figure** — 4 models × 3 CV schemes grouped bar chart |
| `15_confusion_matrices.png`             | per-model holdout confusion matrices |

### Tables (`tables/`)

| Table | Purpose |
|---|---|
| `01_summary_stats.csv`         | per-channel mean / std / min / max (raw and cleaned) |
| `02_outlier_ablation.csv`      | summary stats before vs. after outlier removal |
| `03_cv_accuracy_comparison.csv`| headline 4-model × 3-scheme accuracy table |

## Authors

- **Ivan Del Rio** — [@BariBariGood](https://github.com/BariBariGood)
- **Anish Kondamadugula** — [@arkondamadugula](https://github.com/arkondamadugula)

UC San Diego, COGS 109 — Modelling and Data Analysis, Spring 2026
(Mukamel).

## License

MIT — see [LICENSE](LICENSE).

## References

1. Roesler, O. (2013). *EEG Eye State Data Set*. UCI ML Repository #264.
   <https://archive.ics.uci.edu/dataset/264/eeg+eye+state>
2. Mukamel, R. (2026). *COGS 109 — Modelling and Data Analysis Spring
   2026 Study Guide*. UC San Diego.
3. Roy, Y., Banville, H., Albuquerque, I., et al. (2019). Deep
   learning-based EEG analysis: a systematic review. *J. Neural Eng.*
   16(5):051001.
4. Schirrmeister, R. T., Springenberg, J. T., et al. (2017). Deep
   learning with CNNs for EEG decoding and visualization. *Hum. Brain
   Mapp.* 38(11):5391–5420.
5. James, G., Witten, D., Hastie, T., Tibshirani, R. (2021). *An
   Introduction to Statistical Learning*, 2nd ed. Springer.
6. Bergmeir, C., Benítez, J. M. (2012). On the use of cross-validation
   for time series predictor evaluation. *Information Sciences*
   191:192–213.
