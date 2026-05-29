# Poster — paste-ready content

This is the same content embedded in `poster.pptx`, panel by panel, in
Markdown form. It's here as a fallback in case the `.pptx` import into
Google Slides loses formatting (or if you want to paste sections directly
into a different poster template).

The source-of-truth artifact is `poster.pptx` — regenerate from
`scripts/build_poster.py`.

---

## Title bar

> **Classification of EEG Eye-State via KNN: A Cross-Validation Honesty Study**
>
> Ivan Del Rio • Anish Kondamadugula
> COGS 109 (Mukamel) • Spring 2026
> University of California, San Diego

---

## ABSTRACT / TL;DR

We pick **classification** as our data-analysis approach, **K-nearest
neighbours (KNN)** as our model, and a **k-sweep minimising 5-fold CV
error** as our model-selection procedure. We then run the *same*
procedure under three different CV schemes — shuffled, naive blocked,
and stratified blocked — on the UCI #264 EEG Eye State recording.
All three pick `k = 1`, but the picked accuracies span 47 pp.

> **Shuffled CV says we have a 97% classifier. Honest stratified blocked CV says we have a 78% classifier. The difference is leakage, not signal.**

---

## BACKGROUND & QUESTION

- UCI EEG Eye State dataset (Roesler, 2013) — id #264.
- 14,980 samples × 14 Emotiv channels (AF3, F7, F3, FC5, T7, P7, O1, O2,
  P8, T8, FC6, F4, F8, AF4), sampled at ~128 Hz over ~117 s.
- Single subject, binary `eyeDetection` label.
- Class balance 55.12% open / 44.88% closed → majority baseline 0.5512.
- Only **24 contiguous label runs** in the whole recording — 23 state
  transitions in ~117 seconds.
- Lag-1 channel autocorrelation ≈ 0.997 — adjacent samples are nearly
  identical, so any uniformly-random CV split puts near-duplicates into
  both train and test.

> **Question:** given an autocorrelated single-subject EEG recording and
> a KNN classifier, how does the choice of cross-validation scheme
> change the result of an otherwise-identical model-selection procedure?

---

## METHODS

- **Approach.** Classification (binary `eyeDetection`).
- **Model.** K-nearest neighbours with Euclidean distance in the
  z-scored 14-D channel space. Chosen because KNN's per-sample
  similarity decision rule is the COGS 109-palette model most
  sensitive to lag-1 autocorrelation — exactly what we want for
  studying CV scheme choice.
- **Model selection procedure.** Sweep `k` over the log-spaced grid
  {1, 3, 5, 7, 11, 15, 21, 31, 51, 75, 99, 151, 201}. For each `k` and
  each CV scheme, compute mean 5-fold CV accuracy ± std-dev. The picked
  `k` is the argmax of mean accuracy.
- **Three CV schemes compared:**
  - **shuffled 5-fold** — leaky baseline,
  - **naive blocked 5-fold** — 5 contiguous chunks (no class
    stratification),
  - **stratified blocked 5-fold** — 100 short contiguous segments
    redistributed across folds to balance class proportion (honest).
- **Preprocessing.** Drop 4 voltage outliers (>4 SD on any channel),
  chronological 80/20 split with a 64-sample seam gap, z-score on the
  training partition only.

---

## TAKE-HOME (poster centrepiece)

> **Shuffled CV says we have a 97% classifier. Honest stratified blocked CV says we have a 78% classifier. The difference is leakage, not signal.**

---

## HEADLINE FIGURE

`figures/11_knn_k_sweep.png` — KNN mean accuracy ± std-dev as a function
of `k` (log scale) under all three CV schemes on the same axes.
Shuffled (orange) sits near 0.97 across all `k`; stratified blocked
(green, honest) peaks at 0.778 at `k = 1`; naive blocked (blue) sits
near 0.50 with high variance. All three pick `k = 1`.

---

## RESULTS

KNN model selection under three CV schemes (5-fold each; picked `k` and
picked accuracy at that `k`):

| CV scheme | Picked `k` | Mean accuracy ± std |
|---|---|---|
| Shuffled (leaky baseline) | 1 | 0.9728 ± 0.0037 (≈ 97.3% ± 0.4%) |
| Naive blocked | 1 | 0.5004 ± 0.1068 (≈ 50.0% ± 10.7%) |
| Stratified blocked (honest) | 1 | **0.7778 ± 0.0274 (≈ 77.8% ± 2.7%)** |

Full source: `tables/04_knn_k_sweep.csv` (per-`k` per-scheme numbers)
and `tables/03_cv_accuracy_comparison.csv` (alt-classifier numbers).

---

## SUPPORTING EVIDENCE — alternative classifiers

We also evaluated three alternative classifiers (LDA, PCA→LDA, and
PCR-as-classifier) as sanity checks. Their leakage gaps (shuffled vs
stratified blocked CV) are **+5.6 pp for LDA**, **+2.4 pp for PCA→LDA**,
and **+2.8 pp for PCR-as-classifier** — much smaller than KNN's
**+19.5 pp** gap. This is consistent with KNN being uniquely vulnerable
to lag-1 autocorrelation leakage because it relies on *per-sample*
similarity, while LDA / PCA→LDA / PCR average over many samples before
deciding.

> Supporting figures: `figures/14_cv_comparison_three_way.png` (the
> three-way CV comparison across the four classifiers) and
> `figures/15_confusion_matrices.png` (per-model confusion matrices on
> the chronological holdout test set, ~91% class-0 — interpret with
> care).

---

## AUTOCORRELATION CONTEXT

`figures/08_label_autocorrelation.png` — the binary `eyeDetection`
label autocorrelation across lags 1 through 1000. At lag 1 the
autocorrelation is essentially 1.0; at lag 50 (~0.4 s) it is still
~0.83; it only crosses zero at very long lags. This is the structural
reason shuffled CV leaks on this dataset.

---

## CONCLUSIONS

1. **The same KNN-k-sweep model-selection procedure picks `k = 1` under
   all three CV schemes** — but the accuracy at that `k` varies by
   **47 percentage points** between schemes (50.0% naive blocked → 97.3%
   shuffled).
2. **The honest accuracy at `k = 1` (stratified blocked CV) is
   77.8% ± 2.7%** — well above the 55.12% majority-class baseline.
3. **The 19.5 pp gap** between the shuffled and stratified blocked
   estimates is leakage attributable to scheme choice, *not* signal
   from a better model.
4. **Alternative classifiers (LDA / PCA→LDA / PCR) confirm KNN is
   uniquely vulnerable** — their leakage gaps are only 2–6 pp, exactly
   as theory predicts for classifiers that do not depend on per-sample
   similarity to a single training point.

---

## REFERENCES

1. Roesler, O. (2013). *EEG Eye State Data Set*. UCI ML Repository #264.
   <https://archive.ics.uci.edu/dataset/264/eeg+eye+state>
2. Mukamel, R. (2026). *COGS 109 — Modelling and Data Analysis Spring 2026
   Study Guide*. UC San Diego.
3. James, G., Witten, D., Hastie, T., Tibshirani, R. (2021). *An
   Introduction to Statistical Learning*, 2nd ed. Springer.
4. Bergmeir, C., Benítez, J. M. (2012). On the use of cross-validation
   for time series predictor evaluation. *Information Sciences*
   191:192–213.
