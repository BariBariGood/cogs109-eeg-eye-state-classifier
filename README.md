# EEG Eye-State Classifier — An Honest Cross-Validation Study

> COGS 109 (Spring 2026) final project — UC San Diego — Mukamel.
> Authors: **Ivan Del Rio** ([@BariBariGood](https://github.com/BariBariGood))
> and **Anish Kondamadugula** ([@arkondamadugula](https://github.com/arkondamadugula)).

A small, reproducible project that picks **Classification** as the data
analysis approach, **K-Nearest Neighbors (KNN)** as the model, and a
**hyperparameter sweep over `k` minimising 5-fold cross-validation
error** as the model selection procedure — and then runs that *same*
procedure under three different CV schemes (shuffled, naive blocked,
stratified blocked) to show how much the scheme choice changes the
answer on autocorrelated EEG data.

The headline number is that the picked model, **KNN with k = 1**,
reports **77.8% ± 2.7%** accuracy under the honest stratified blocked
CV scheme — well above the 55.12% majority-class baseline — but the
*same* `k = 1` reports **97.3% ± 0.4%** under leaky shuffled CV. The
**19.5 percentage-point gap** is leakage attributable to scheme choice,
not model quality.

The full paper-style writeup lives in [`report/`](report/) — a LaTeX
research paper [`report/final_report.tex`](report/final_report.tex)
(built PDF: [`report/final_report.pdf`](report/final_report.pdf), ~10
pages) alongside the original Markdown draft
[`report/final_report.md`](report/final_report.md) — and the academic
poster lives at [`poster/poster.pptx`](poster/poster.pptx).

## Executive summary

> **Shuffled CV says we have a 97% classifier. Honest stratified blocked CV says we have a 78% classifier. The difference is leakage, not signal.**

## Hypothesis & contribution

**Hypothesis.** A K-Nearest Neighbors classifier trained on raw
14-channel EEG voltages can predict per-sample eye state above the
55.12% majority-class baseline, but accuracy estimates obtained under
shuffled k-fold cross-validation will substantially exceed honest
out-of-time estimates because of temporal leakage between adjacent
samples.

**Contribution.** The contribution of this project is *methodological*
rather than predictive. We perform the same KNN-k-sweep model selection
procedure (sweep `k` over a log-spaced grid; pick the `k` that minimises
mean 5-fold CV error) under three different CV schemes — shuffled,
naive blocked, and stratified blocked — on the UCI #264 EEG Eye State
dataset, and document how the picked `k` and the picked accuracy change
between schemes. We then provide a small sanity-check appendix that
runs the same model selection idea over three alternative classifiers
(LDA, PCA→LDA, PCR-as-classifier) to confirm that the leakage problem
is specifically a KNN problem on this dataset and not a generic
modelling failure.

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
so a classifier that exploits sample-to-sample similarity (KNN at
small `k`) trivially recovers it from a shuffled training fold.

## Methodology (short version)

The full methodology lives in
[`report/final_report.md`](report/final_report.md). In short:

1. **Preprocessing.** Drop 4 outlier samples (any channel > 4 SD
   above mean), chronologically split 80/20 train/test with a
   64-sample seam gap, z-score channels using train-only statistics.
2. **Approach.** Classification (binary eye-state label).
3. **Model.** K-Nearest Neighbors. KNN is the simplest non-parametric
   classifier in the COGS 109 palette, makes no distributional
   assumption, and — because it predicts from per-sample similarity —
   naturally exposes the lag-1 EEG autocorrelation that drives the
   leakage gap we are studying.
4. **Model selection procedure.** Sweep `k ∈ {1, 3, 5, 7, 11, 15, 21,
   31, 51, 75, 99, 151, 201}` (log-spaced). For each `k` and each CV
   scheme, compute mean 5-fold CV accuracy ± std-dev. The picked `k`
   for a scheme is the argmin of mean CV error (equivalently argmax
   of mean CV accuracy).
5. **Three CV schemes evaluated**, all `k = 5`:
   - **shuffled** — uniform-random split (leaky baseline);
   - **naive blocked** — 5 contiguous time chunks (leakage-resistant
     but class-imbalanced per fold on this single-subject recording);
   - **stratified blocked** — 100 short contiguous segments
     redistributed across folds to balance class proportion.
6. **Headline result.** Figure 11 (`figures/11_knn_k_sweep.png`) shows
   accuracy vs `k` under each scheme. Shuffled picks `k = 1` at 97.3%,
   naive blocked picks `k = 1` at 50.0%, and stratified blocked picks
   `k = 1` at 77.8%. The same procedure, the same model, the same
   data — three different answers.
7. **Sanity-check appendix.** We also ran the same per-model
   hyperparameter sweep on three alternative classifiers from the
   COGS 109 palette — LDA (no tuning), PCA→LDA (sweep `n_components`),
   and PCR-as-classifier (sweep `n_components`) — and report their
   leakage gaps as supporting evidence. Their results are summarised
   in figure 14 and detailed in the modelling notebook.

No spectral / FFT features, no logistic regression, no SVM, no decision
trees, no neural networks — only the COGS 109 study-guide methods.

## Headline results

KNN model selection under three CV schemes (5-fold each, log-spaced
`k` grid; per-scheme picked `k` shown):

| CV scheme | Picked `k` | Mean accuracy ± std |
|---|---|---|
| Shuffled (leaky baseline) | 1 | 0.9728 ± 0.0037 (≈ 97.3% ± 0.4%) |
| Naive blocked | 1 | 0.5004 ± 0.1068 (≈ 50.0% ± 10.7%) |
| Stratified blocked (honest) | 1 | **0.7778 ± 0.0274 (≈ 77.8% ± 2.7%)** |

> **Shuffled CV says we have a 97% classifier. Honest stratified blocked CV says we have a 78% classifier. The difference is leakage, not signal.**

The 19.5 percentage-point gap between the leaky shuffled estimate
(97.3%) and the honest stratified blocked estimate (77.8%) is the
**leakage attributable to scheme choice**, not to model quality —
exactly the same KNN, exactly the same training partition, exactly the
same `k`, only the cross-validation rule changes.

Headline figure:

![KNN model selection under three CV schemes](figures/11_knn_k_sweep.png)

**Supporting evidence from alternative classifiers.** As a sanity
check we also ran a hyperparameter sweep on three other models from
the COGS 109 palette — LDA, PCA→LDA (sweep `n_components`), and
PCR-as-classifier (sweep `n_components`) — under all three CV schemes.
The leakage gaps (shuffled minus stratified blocked) for the picked
hyperparameter were **+5.6 pp for LDA**, **+2.4 pp for PCA→LDA**, and
**+2.8 pp for PCR-as-classifier**, compared to KNN's **+19.5 pp** gap.
This is consistent with KNN being uniquely vulnerable to lag-1
autocorrelation leakage because it relies on *per-sample* similarity,
while LDA / PCA→LDA / PCR average over many samples before deciding.
The full alternative-classifier numbers live in
[`figures/14_cv_comparison_three_way.png`](figures/14_cv_comparison_three_way.png)
and [`tables/03_cv_accuracy_comparison.csv`](tables/03_cv_accuracy_comparison.csv).

## Repository layout

```
.
├── data/
│   ├── raw/                       # raw CSV pulled from UCI + manifest
│   └── processed/                 # outlier-cleaned, z-scored splits + CV folds
├── figures/                       # 15 committed PNG figures (EDA + modelling)
├── tables/                        # CSV summary tables (incl. k-sweep + 3-way CV)
├── notebooks/
│   ├── 00_fetch_data.ipynb        # downloads UCI #264, writes idempotent manifest
│   ├── 01_eda.ipynb               # exploratory analysis (11 figures, 2 tables)
│   └── 02_modeling.ipynb          # KNN k-sweep + 3-way CV + alt-classifier appendix
├── poster/
│   ├── poster.pptx                # 48"×36" final poster (source of truth)
│   └── README.md                  # poster-specific notes
├── report/
│   ├── final_report.tex           # LaTeX research paper (COGS 109 rubric sections)
│   ├── final_report.pdf           # built PDF (~10 pages)
│   └── final_report.md            # original Markdown draft (~4000 words)
├── scripts/
│   ├── preprocess.py              # full preprocessing CLI (idempotent)
│   ├── regenerate_figure_11.py    # standalone KNN k-sweep figure (3 CV schemes)
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
make poster      # regenerate poster/poster.pptx
pytest tests/    # 18 tests, < 2 min
```

To regenerate the headline figure on its own without re-running the
full modelling notebook:

```bash
python scripts/regenerate_figure_11.py
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
| `make poster`   | run `scripts/build_poster.py` (writes `poster/poster.pptx`)|
| `make report`   | build `report/final_report.pdf` from LaTeX (needs `pdflatex`)|
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
| `notebooks/02_modeling.ipynb`     | KNN k-sweep + 3-way CV comparison + alternative-classifier appendix          |

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
| `11_knn_k_sweep.png`                    | **headline figure** — KNN k-sweep under 3 CV schemes |
| `12_pca_lda_components_sweep.png`       | PCA→LDA accuracy vs. number of components (alt-classifier) |
| `13_pcr_components_sweep.png`           | PCR accuracy vs. number of components (alt-classifier) |
| `14_cv_comparison_three_way.png`        | supporting figure — 4 models × 3 CV schemes grouped bar chart |
| `15_confusion_matrices.png`             | per-model holdout confusion matrices |

### Tables (`tables/`)

| Table | Purpose |
|---|---|
| `01_summary_stats.csv`         | per-channel mean / std / min / max (raw and cleaned) |
| `02_outlier_ablation.csv`      | summary stats before vs. after outlier removal |
| `03_cv_accuracy_comparison.csv`| 4-model × 3-scheme accuracy table (alt-classifier reference) |
| `04_knn_k_sweep.csv`           | KNN accuracy ± std for every `k` under every CV scheme |

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
3. James, G., Witten, D., Hastie, T., Tibshirani, R. (2021). *An
   Introduction to Statistical Learning*, 2nd ed. Springer.
4. Bergmeir, C., Benítez, J. M. (2012). On the use of cross-validation
   for time series predictor evaluation. *Information Sciences*
   191:192–213.
