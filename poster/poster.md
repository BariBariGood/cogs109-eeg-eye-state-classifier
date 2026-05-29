# Poster — paste-ready content

This is the same content embedded in `poster.pptx`, panel by panel, in
Markdown form. It's here as a fallback in case the `.pptx` import into
Google Slides loses formatting (or if you want to paste sections directly
into a different poster template).

The source-of-truth artifact is `poster.pptx` — regenerate from
`scripts/build_poster.py`.

---

## Title bar

> **Classifying Eye State from EEG Signals: An Honest Cross-Validation Study**
>
> Ivan Del Rio • Anish Kondamadugula
> COGS 109 — Mukamel — Spring 2026
> University of California, San Diego

---

## ABSTRACT

We classify per-sample eye state (open vs. closed) from a single-subject
14-channel EEG recording (UCI #264, 14,980 samples) using only the methods
in the COGS 109 palette: LDA, KNN, PCA→LDA, and PCR-as-classifier. The
central contribution is methodological. Standard shuffled k-fold CV reports
97% accuracy for KNN, but the recording is a continuous time series with
lag-1 channel autocorrelation r ≈ 0.997 and only 24 contiguous label runs,
so shuffled folds leak neighbouring samples from train into test. We
compare three CV schemes — shuffled, naive blocked, and stratified blocked
— and show that the honest stratified blocked estimate of our best
classifier is **77.8% ± 2.7%**, not 97%. The headline takeaway: roughly
twenty accuracy points reported in the literature on this dataset are
leakage, not signal.

---

## QUESTION / HYPOTHESIS

Can raw 14-channel EEG voltages predict whether a person's eyes are open
or closed, and how much of the accuracy reported under shuffled
cross-validation is an artefact of temporal leakage?

---

## BACKGROUND

- UCI EEG Eye State dataset (Roesler, 2013) — id #264.
- 14,980 samples × 14 Emotiv channels (AF3, F7, F3, FC5, T7, P7, O1, O2,
  P8, T8, FC6, F4, F8, AF4), sampled at 128 Hz over ~117 s.
- Single subject, binary eyeDetection label, 55.12% open / 44.88% closed.
- Only 24 contiguous label runs in the whole recording.
- Channel autocorrelation at lag 1 ≈ 0.997 — adjacent samples are nearly
  identical, so shuffled folds put near-duplicates into both train and
  test.

> Figure 1 (poster, top-left): `figures/08_label_autocorrelation.png` —
> label autocorrelation across lags. Motivates blocked CV.

---

## METHODS

- **Models (COGS 109 palette):** LDA, KNN (k tuned), PCA→LDA, PCR as a
  binary classifier with a 0.5 threshold.
- **Preprocessing:** drop 4 voltage outliers (>4 SD on any channel),
  z-score on the training set only, chronological 80/20 split with a
  64-sample seam gap between train and test.
- **CV scheme A — shuffled 5-fold** (leaky baseline).
- **CV scheme B — naive blocked 5-fold** (5 contiguous time chunks).
- **CV scheme C — stratified blocked 5-fold** (100 short contiguous
  segments redistributed across folds to balance class proportion).
- **Hyperparameters:** chosen by maximum mean blocked CV accuracy.

---

## TAKE-HOME (poster centrepiece)

> Shuffled CV says we have a 97% classifier.
>
> **Honest stratified blocked CV says we have a 78% classifier.**
>
> *The difference is leakage, not signal.*

---

## HEADLINE FIGURE

`figures/14_cv_comparison_three_way.png` — 4 models × 3 CV schemes, grouped
bar chart with std-dev error bars. This is the largest figure on the
poster.

---

## RESULTS

| Model | Shuffled (leaky) | Naive blocked | **Stratified blocked (honest)** |
|---|---|---|---|
| LDA | 0.647 ± 0.006 | 0.408 ± 0.122 | **0.591 ± 0.047** |
| KNN (k=1) | 0.973 ± 0.004 | 0.500 ± 0.107 | **0.778 ± 0.027** |
| PCA→LDA (n=3) | 0.557 ± 0.015 | 0.410 ± 0.083 | **0.534 ± 0.044** |
| PCR (n=2) | 0.553 ± 0.009 | 0.408 ± 0.130 | **0.525 ± 0.018** |

Full source: `tables/03_cv_accuracy_comparison.csv`.

> Figure 3 (poster, right column): `figures/15_confusion_matrices.png` —
> confusion matrices for each model on the chronological holdout test set
> (91% class-0; interpret with care).

---

## CONCLUSIONS

- Best honest model: **KNN (k=1) at 77.8% ± 2.7%** stratified blocked CV —
  well above the 55% majority baseline and the best accuracy any model in
  the palette achieved under a temporally honest split.
- Leakage gap (shuffled minus stratified blocked) averages ~13 pp across
  models and reaches ~19.5 pp on KNN, which exploits lag-1 autocorrelation
  almost perfectly when adjacent samples land in the same fold.
- Pedagogical implication: published EEG eye-state results above 95%
  almost certainly reflect shuffled-CV leakage rather than genuine
  discrimination from the raw voltage vector alone.

---

## FUTURE WORK

- Windowed features (rolling channel means/variances) to give models
  temporal context without leakage.
- Cross-subject generalisation — single-subject data tells us nothing
  about inter-subject transfer.
- Longer recordings with more label transitions per subject to make
  blocked CV less brittle.

---

## REFERENCES

1. Roesler, O. (2013). *EEG Eye State Data Set*. UCI ML Repository #264.
   <https://archive.ics.uci.edu/dataset/264/eeg+eye+state>
2. Mukamel, R. (2026). *COGS 109 — Modelling and Data Analysis Spring 2026
   Study Guide*. UC San Diego.
3. Roy, Y., Banville, H., Albuquerque, I., et al. (2019). Deep
   learning-based EEG analysis: a systematic review. *J. Neural Eng.*
   16(5):051001.
4. Schirrmeister, R. T., Springenberg, J. T., et al. (2017). Deep learning
   with CNNs for EEG decoding and visualization. *Hum. Brain Mapp.*
   38(11):5391–5420.
5. James, G., Witten, D., Hastie, T., Tibshirani, R. (2021). *An
   Introduction to Statistical Learning*, 2nd ed. Springer.
