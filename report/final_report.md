# Classifying Eye State from EEG Signals via KNN: A Cross-Validation Honesty Study

**Ivan Del Rio** &nbsp;·&nbsp; **Anish Kondamadugula**
COGS 109 — Modelling and Data Analysis &nbsp;·&nbsp; Mukamel &nbsp;·&nbsp; Spring 2026
University of California, San Diego

---

## Abstract

We pick a single data-analysis approach from the COGS 109 SP26 study
guide — **classification** — and a single model from within that
approach — **K-nearest neighbours (KNN)** — and we perform model
selection by sweeping the hyperparameter `k` over a log-spaced grid
{1, 3, 5, 7, 11, 15, 21, 31, 51, 75, 99, 151, 201} to find the value
that minimises 5-fold cross-validation error on the UCI #264 EEG Eye
State dataset (Roesler, 2013). We then run that *same* model-selection
procedure under three different cross-validation schemes: shuffled,
naive blocked, and stratified blocked. The methodological contribution
of the report is to document how scheme choice changes the picked
hyperparameter and the picked accuracy on an autocorrelated EEG
recording. All three schemes pick `k = 1` as optimal, but the picked
accuracies differ dramatically: **97.3% ± 0.4%** under shuffled CV,
**50.0% ± 10.7%** under naive blocked CV, and **77.8% ± 2.7%** under
the honest stratified blocked CV. The **19.5 percentage-point gap**
between shuffled and stratified blocked accuracy is leakage
attributable to scheme choice — not to model quality, since the same
KNN model is evaluated in all three settings on the same training
data. A small sanity-check appendix runs the same idea on three
alternative classifiers (LDA, PCA→LDA, PCR-as-classifier) and shows
their leakage gaps are only 2–6 pp, consistent with KNN being
uniquely vulnerable to lag-1 autocorrelation because it predicts from
per-sample similarity.

> **Shuffled CV says we have a 97% classifier. Honest stratified blocked CV says we have a 78% classifier. The difference is leakage, not signal.**

---

## 1. Introduction

EEG eye-state classification on the UCI #264 dataset is one of the
cleanest small-scale settings in which to study cross-validation
honesty. Online tutorials and student write-ups routinely report KNN
accuracies in the 95–98% range on this exact dataset under standard
shuffled k-fold cross-validation. We argue in this report that those
numbers reflect a methodological artefact rather than a strong
classifier. The dataset is a single ~117-second continuous recording
from one subject; its label changes only 23 times across the 14,980
samples; adjacent samples are nearly identical (lag-1 channel
correlation ≈ 0.997); and the label is piecewise-constant over runs
that are sometimes thousands of samples long. A KNN classifier with
`k = 1` evaluated under shuffled k-fold CV is essentially asking "what
was the eye state of the sample 8 ms before / after this one?" — a
question to which it can almost always answer correctly. That is a
property of the CV rule, not of the classifier.

We framed the project around the assignment template that the
COGS 109 SP26 course requires: **approach** = classification,
**model** = K-nearest neighbours, **model selection procedure** = sweep
`k` over a log-spaced grid and pick the value that minimises mean
5-fold CV error. Our methodological contribution is to run that *same*
procedure under three different CV schemes — shuffled, naive blocked,
stratified blocked — and document how the answer changes.

The remainder of the report is organised as follows. §2 describes the
dataset, label structure, and autocorrelation that motivate blocked
CV. §3 specifies preprocessing, the classification approach, the KNN
model, the k-sweep procedure, the three CV schemes, and a small
sanity-check appendix on three alternative classifiers. §4 presents
the headline figure, per-scheme picked `k` and picked accuracy, the
leakage-gap analysis, and supporting alternative-classifier evidence.
§5 discusses why KNN is uniquely vulnerable. §6 lists limitations and
§7 concludes.

---

## 2. Dataset

The UCI EEG Eye State dataset (Roesler, 2013; UCI Machine Learning
Repository #264) is a single uninterrupted recording from one subject
wearing an Emotiv EPOC consumer-grade EEG headset. The data consist of
14 voltage channels — AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6,
F4, F8, AF4 — sampled at ~128 Hz over approximately 117 seconds, plus
a binary `eyeDetection` label (0 = eyes open, 1 = eyes closed)
annotated by the experimenter. The total length is 14,980 samples.

The class balance is 55.12% eyes-open (8,257 samples) and 44.88%
eyes-closed (6,723 samples), giving a majority-class baseline of
**0.5512** — the natural floor any classifier must beat.

![Figure 1: Class balance — 55.12% open vs. 44.88% closed.](../figures/01_class_balance.png)
*Figure 1.  Class balance of the UCI #264 EEG eye-state dataset. The
majority class (eyes open) sets the random-guess baseline at 0.5512.*

A purely structural feature of this dataset deserves special
attention. The labels do not change frequently: the 14,980-sample
recording contains only **24 contiguous label runs** — that is, the
subject opens or closes their eyes only 23 times during the entire
~117 seconds. Runs vary substantially in length, from fewer than a
hundred samples up to several thousand consecutive samples. Figure 2
shows a one-second window of the raw signal with the label strip
overlaid.

![Figure 2: Raw EEG with eye-state strip overlay.](../figures/04_timeseries_with_label.png)
*Figure 2.  Raw 14-channel EEG voltages (top) with the eye-state label
strip (bottom). Channels are continuous and smooth on the millisecond
scale; the label is piecewise-constant over multi-second runs.*

The structural consequence is severe. Lag-1 autocorrelation of any
single voltage channel is approximately 0.997 — adjacent samples are
nearly identical. The label, being piecewise-constant over long
runs, has very high short-lag autocorrelation: from Figure 3 it is
~1.0 at lag 1, ~0.83 at lag 50 (≈ 0.4 s), and only crosses zero at
very long lags.

![Figure 3: Eye-state label autocorrelation across lags.](../figures/08_label_autocorrelation.png)
*Figure 3.  Autocorrelation of the binary `eyeDetection` label across
lags 1 through 1000. The label is essentially identical to itself for
the first ~100 samples (~0.8 seconds) and only decorrelates at very
long lags. This is the structural reason shuffled CV leaks: a
sample's nearest neighbour in time is almost always in its own class.*

This is the structural problem at the heart of the report: any CV
protocol that splits samples uniformly at random leaves each test
sample with a near-duplicate in its own training set. A model whose
prediction depends on per-sample similarity — KNN with `k = 1` being
the canonical example — will appear spectacular, but the score
reflects memorisation, not a generalisable voltage-to-label rule.

---

## 3. Methodology

### 3.1 Preprocessing

Outliers in the raw signal — four samples with at least one channel
exceeding 4 standard deviations above the channel mean (z > 4) — were
removed prior to any further processing. This is a standard EEG
hygiene step and accounts for occasional headset disconnections or
muscle artefacts; an ablation reported in
`tables/02_outlier_ablation.csv` shows it does not materially change
downstream model accuracy but does stabilise per-channel summary
statistics.

We then split the cleaned recording chronologically into 80% train
(samples 1 – 11,981, n = 11,917 after outlier removal) and 20% test
(samples 12,046 – 14,976, n = 2,995), inserting a **64-sample seam
gap** (0.5 seconds at 128 Hz) between train and test to guarantee no
train sample is within one half-second of a test sample. This is the
minimum cross-fold isolation needed given the autocorrelation profile
in Figure 3, where label autocorrelation remains > 0.5 out to about
lag 200; we chose 64 as a conservative compromise between isolation
and the cost of throwing away contiguous data.

After the split we z-scored each channel using only the **training**
mean and standard deviation. The fitted scaler parameters are
persisted to `data/processed/scaler.json` and reused for test
evaluation. Z-scoring before the split would itself be a form of
leakage (the test set would shape the channel scale), so we explicitly
defer it to after the split.

### 3.2 Approach selection — classification

The COGS 109 SP26 course covers several broad approaches
(linear / polynomial / spline regression, PCR, LDA, KNN, K-means,
hierarchical clustering, PCA). The target variable here is the binary
`eyeDetection` label, so **classification** is the natural approach:
we want a model that maps a 14-dimensional voltage vector to one of
two discrete labels. Regression approaches do not target a discrete
label without artificial thresholding; clustering approaches do not
use the label at all. We therefore restrict ourselves to the
classification methods in the palette.

### 3.3 Model selection — K-nearest neighbours

Within classification we picked **KNN** as our model. KNN is the
simplest non-parametric classifier in the palette — it makes no
distributional assumption and has exactly one hyperparameter (`k`)
that controls the bias–variance tradeoff, making it an ideal
candidate for a clean single-axis hyperparameter sweep. More
importantly, because KNN predicts from per-sample similarity to the
training set, it is the model in the palette that *most directly*
exploits the lag-1 autocorrelation structure of EEG data. If we want
to study how CV scheme choice changes a model-selection result,
picking the model most sensitive to that choice gives the cleanest
empirical signal. KNN with small `k` is also competitive with the
reported literature numbers (~97% under shuffled CV) on this dataset,
so the leakage gap we measure is a comparison between an honest and a
leaky estimate of the same strong baseline.

### 3.4 Model selection procedure — k-sweep

The hyperparameter to tune for KNN is `k`, the number of nearest
neighbours that vote on the test sample's label. We sweep `k` over the
log-spaced grid

```
k ∈ {1, 3, 5, 7, 11, 15, 21, 31, 51, 75, 99, 151, 201}
```

The grid is odd-only (avoiding ties on the binary label), starts at
`k = 1` (the most leakage-prone setting), and extends to `k = 201` (a
local average that should damp out lag-1 sample similarity if the
classification signal is real).

For each `k` and each CV scheme, we run 5-fold cross-validation on the
z-scored training partition. We then aggregate per-fold accuracy as
the mean ± std-dev across folds. The **picked `k`** under a given
scheme is the argmin of mean CV error (equivalently the argmax of
mean CV accuracy). Re-running the same `k` grid under a different CV
scheme gives a different per-`k` accuracy curve, but the
argmax-of-accuracy rule is identical across schemes — *the
model-selection procedure itself does not change*.

### 3.5 Cross-validation schemes

We compare three 5-fold cross-validation schemes on the training set:

- **Scheme A — shuffled 5-fold.** Samples are uniformly shuffled and
  split into 5 equal folds. This is the standard `KFold(shuffle=True)`
  protocol and the baseline most students of this dataset run by
  default. It is the leakiest possible scheme on a continuous time
  series.
- **Scheme B — naive blocked 5-fold.** The training set is split into
  5 contiguous time chunks. No shuffling is performed. This
  eliminates most lag-≤1 leakage but introduces a new problem:
  because the label runs are long and unevenly distributed,
  individual folds can end up near-class-pure (e.g. one fold lands at
  ~20% class-0 vs. the training partition's ~46.5% class-0). The per-fold accuracy
  variance therefore explodes, and the mean accuracy can fall *below*
  the majority-class baseline. We include this scheme primarily to
  make the failure mode visible — it is *not* a scheme we would
  recommend in practice.
- **Scheme C — stratified blocked 5-fold.** This is the protocol we
  recommend as the honest evaluator. We split the training set into
  100 short contiguous segments (each ~120 samples ≈ 0.9 seconds
  long), preserving within-segment temporal continuity. Segments are
  then assigned to folds in a class-stratified round-robin so each
  fold has approximately the same eyes-open / eyes-closed proportion
  as the full training set. The segment length is short enough that
  the per-segment label is nearly always constant, and long enough
  that most segments do not straddle a label transition. This scheme
  keeps the temporal-isolation guarantees of blocked CV while
  restoring the class-balance guarantees of stratified CV.

See `figures/04b_timeseries_with_folds.png` for a visualisation of the
per-fold sample assignments under all three schemes.

### 3.6 Alternative classifiers (sanity checks)

To check whether the leakage gap we will see for KNN is a
KNN-specific phenomenon or a generic problem with classification on
this dataset, we ran the same model-selection idea on three
alternative classifiers from the COGS 109 palette. **LDA** has no
tunable hyperparameter — we report a single accuracy per scheme.
**PCA → LDA** sweeps `n ∈ {1, 2, 3, 5, 7, 10}` and picks the `n`
maximising mean stratified blocked CV accuracy. **PCR-as-classifier**
(linear regression on the first `n` principal components, thresholded
at 0.5) sweeps the same `n` grid. These three are supporting evidence,
not the primary result; their numbers live in
`tables/03_cv_accuracy_comparison.csv` and
`figures/14_cv_comparison_three_way.png`.

### 3.7 Metrics

We report:

- **Accuracy** = (TP + TN) / N — the headline metric used throughout
  for both CV and holdout.
- **Sensitivity (recall)** and **specificity** — used per-model in
  the confusion matrices on the chronological holdout (Figure 5).

The chronological holdout test partition is heavily class-imbalanced
(90.7% class-0) because the very last segment of the recording
happens to be mostly eyes-open. We do not over-interpret holdout
accuracy in isolation; the per-fold CV numbers in §4 are the more
meaningful summary.

---

## 4. Results

### 4.1 KNN model selection under three CV schemes

Figure 4 is the headline figure of the report. It plots KNN mean
accuracy ± std-dev as a function of `k` (log scale) under all three
CV schemes on the same axes.

![Figure 4: KNN k-sweep under three CV schemes.](../figures/11_knn_k_sweep.png)
*Figure 4.  KNN model selection under three CV schemes (5-fold each;
error bars are ±1 SD across folds). Orange squares = shuffled CV
(leaky baseline). Blue circles = naive blocked CV. Green diamonds =
stratified blocked CV (honest). The horizontal dashed line is the
majority-class baseline (0.5512). The small open marker on each curve
highlights the argmax-accuracy `k` for that scheme together with the
picked accuracy.*

Three qualitative observations are immediate. The shuffled curve sits
near 0.97 across all `k`, decaying roughly linearly on the log axis
from 0.973 at `k = 1` to ~0.84 at `k = 201` — essentially "nearly
perfect" everywhere. The naive blocked curve sits near 0.50 across
all `k`, at or below the majority baseline, with a very large per-fold
standard deviation (~10–13 pp); this is fold construction failing,
not the model failing (see §3.5 and §5). The stratified blocked
curve peaks at 0.778 at `k = 1` and drops gracefully to ~0.71 by
`k = 201`, with a much smaller std-dev (~2.7–4.0 pp).

### 4.2 Per-scheme picked `k` and picked accuracy

Applying the argmax-of-accuracy model-selection rule (§3.4) to each
scheme yields:

| CV scheme | Picked `k` | Picked accuracy ± std |
|---|---|---|
| Shuffled (leaky baseline) | 1 | 0.9728 ± 0.0037 (≈ 97.3% ± 0.4%) |
| Naive blocked | 1 | 0.5004 ± 0.1068 (≈ 50.0% ± 10.7%) |
| Stratified blocked (honest) | 1 | **0.7778 ± 0.0274 (≈ 77.8% ± 2.7%)** |

*Table 1.  Per-scheme picked `k` and picked accuracy from the KNN
k-sweep. All three schemes pick `k = 1` — small `k` always wins on
this autocorrelated single-subject recording, because per-sample
similarity carries the discriminative signal whether or not that
signal is leakage. The picked accuracies, however, span 47
percentage points.*

The remarkable result is that **all three schemes pick the same
hyperparameter** (`k = 1`). The model-selection procedure is
robust — the *answer* it gives is not. The picked accuracy ranges
from 50.0% (naive blocked) to 97.3% (shuffled), a spread of 47
percentage points, despite the procedure, the model, the data, and
the picked `k` being identical.

### 4.3 Leakage-gap analysis

The methodological centrepiece of the report is the **leakage gap**:
the difference between an estimator's leaky shuffled-CV accuracy and
its honest stratified blocked-CV accuracy at the same picked
hyperparameter. For KNN at `k = 1` this is

```
leakage gap = 0.9728 − 0.7778 = 0.1950 (≈ 19.5 pp).
```

We attribute this 19.5 pp gap to leakage of adjacent samples across
the train/test fold boundary, not to model quality. The model is
KNN with `k = 1` in both cases; the training partition is the same;
the only thing that changes is the rule that assigns samples to
folds. Under shuffled CV the temporal neighbour of every test sample
lands in the training fold with ~80% probability and carries the
same label with ~99.7% probability; under stratified blocked CV the
temporal neighbour is reserved into the same segment as the test
sample, and the next-nearest training sample is potentially seconds
away from a label transition.

The honest accuracy at `k = 1` is **77.8% ± 2.7%** — well above the
55.12% majority-class baseline, by ~23 percentage points. The
classifier is not useless; it is just much less impressive than the
shuffled-CV number implies.

### 4.4 Supporting evidence — alternative classifiers

To rule out the possibility that the leakage gap we see for KNN is a
generic property of classification on this dataset, we ran the same
model-selection idea on three alternative classifiers from the
COGS 109 palette. Their picked-hyperparameter leakage gaps are
summarised below; the full per-scheme numbers live in
`tables/03_cv_accuracy_comparison.csv`.

| Model | Picked hyperparameter | Shuffled CV | Stratified blocked CV | Leakage gap |
|---|---|---|---|---|
| LDA | n/a (no sweep) | 0.6471 ± 0.0063 | 0.5911 ± 0.0474 | +5.6 pp |
| PCA → LDA | n = 3 | 0.5574 ± 0.0147 | 0.5339 ± 0.0438 | +2.4 pp |
| PCR (as classifier) | n = 2 | 0.5528 ± 0.0087 | 0.5250 ± 0.0176 | +2.8 pp |
| **KNN** (headline) | **k = 1** | **0.9728 ± 0.0037** | **0.7778 ± 0.0274** | **+19.5 pp** |

*Table 2.  Per-model leakage gaps (shuffled CV minus stratified
blocked CV) at the picked hyperparameter. KNN's leakage gap is the
outlier: ~4–8x larger than the gap for any other classifier in the
palette.*

![Figure 5: Three-way CV comparison across the four classifiers.](../figures/14_cv_comparison_three_way.png)
*Figure 5.  Supporting figure — three-way 5-fold CV accuracy across
the four COGS 109-palette classifiers. Orange = shuffled, blue =
naive blocked, green = stratified blocked. The KNN bar group has the
largest shuffled-to-stratified-blocked gap, consistent with KNN being
uniquely vulnerable to the lag-1 autocorrelation that drives the
leakage problem.*

The pattern is exactly what we would expect if KNN's leakage gap
were caused by per-sample autocorrelation:

- LDA's decision rule is a single global linear projection of the
  14-D feature space onto a 1-D axis; it cannot exploit per-sample
  similarity, so its leakage gap is small (5.6 pp).
- PCA → LDA and PCR both project first to a low-dimensional subspace
  before classification, which destroys most of the high-frequency
  sample-to-sample similarity that KNN exploits; their leakage gaps
  are even smaller (2.4 pp and 2.8 pp).
- KNN at `k = 1` is the only classifier whose prediction depends on
  the identity of the nearest training sample. Its leakage gap is
  4–8x larger than the next-largest in the palette.

We treat this as supporting evidence rather than the primary
result — the headline of the report is the KNN k-sweep under three
CV schemes, not the four-model comparison.

### 4.5 Holdout test set

For completeness, Figure 6 shows the per-model confusion matrices on
the final 20% chronological holdout test set.

![Figure 6: Confusion matrices for each model on the holdout test.](../figures/15_confusion_matrices.png)
*Figure 6.  Per-model confusion matrices on the 20% chronological
holdout test set. The holdout is dominated by eyes-open samples
(~91% class-0), so accuracy on the holdout is mostly driven by
specificity. The number in the top-right of each panel is the
per-model accuracy on this set.*

The holdout's 90.7% class-0 skew is a quirk of the recording — the
last 20% happens to fall mostly inside a single long eyes-open run.
We did not rebalance the holdout because doing so would itself
introduce leakage. The CV numbers in §4.2 and §4.3 are the more
meaningful summary; we report holdout accuracy only as a confirmatory
check.

---

## 5. Discussion

### 5.1 Why is KNN uniquely vulnerable?

A 1-NN classifier predicts the test sample's class as the class of
its single nearest training neighbour. On this dataset, because the
lag-1 channel correlation is ~0.997, **the nearest neighbour in
feature space is almost always the temporal neighbour**. Under
shuffled CV that temporal neighbour is in the training fold ~80% of
the time and shares the test sample's class ~99.7% of the time
(because the label is piecewise-constant over multi-second runs).
Combine the two and KNN with `k = 1` trivially achieves > 95%
accuracy without learning anything about voltage-to-eye-state mapping.

Under stratified blocked CV the temporal neighbour is reserved into
the same segment as the test sample, so KNN must fall back to the
next-nearest *non-adjacent* sample — potentially seconds away from a
label transition. KNN's accuracy drops to 77.8%.

This is the take-home message:

> Shuffled CV says we have a 97% classifier. Honest stratified blocked CV says we have a 78% classifier. The difference is leakage, not signal.

The difference is not about which model we picked, or whether we
swept the hyperparameter carefully enough, or whether we standardised
the channels correctly. The model-selection procedure is identical in
all three settings. The difference is entirely about whether the
cross-validation rule allows the temporal neighbour of each test
sample to leak into the training fold.

### 5.2 Why does naive blocked CV underperform the majority baseline?

The 5 contiguous chunks of the training set are not class-balanced.
Because the label changes only 23 times in the whole recording,
several of the chunks straddle very few or zero label transitions,
and the class proportion within a chunk can be very far from the
global 55% / 45% balance. A classifier trained on four chunks that
happen to contain mostly eyes-open then predicts mostly eyes-open on
the held-out chunk, which might be mostly eyes-closed. The result is
an accuracy *below* the majority baseline. This is not a model
failure but a fold-stratification failure: per-fold class imbalance
dominates per-fold accuracy when `k_folds = 5` and the underlying
label is so blocky. Stratifying the segments while still keeping each
segment contiguous fixes the problem, which is why stratified
blocked CV recovers from the naive blocked failure mode.

### 5.3 Implication for reported accuracies in the literature

A non-trivial fraction of student projects and tutorial notebooks on
this dataset report KNN accuracies in the 95–98% range using shuffled
k-fold. Our analysis suggests those numbers are an artefact of the CV
protocol rather than a powerful classifier: the honest KNN accuracy
is 77.8%, and the extra 20 pp are leakage of adjacent samples between
train and test folds. This is consistent with broader cautions in
the time-series-CV literature about i.i.d. CV being inappropriate for
autocorrelated signals (Bergmeir & Benítez, 2012).

---

## 6. Limitations and Future Work

- **Single-subject dataset.** UCI #264 is one subject; we cannot tell
  whether the findings generalise across people. Cross-subject
  transfer is a known hard problem in EEG.
- **Consumer EEG hardware.** The Emotiv EPOC is a 14-channel
  consumer-grade headset; we do not characterise per-channel signal
  quality.
- **Only ~117 seconds of data.** A single short recording. Longer
  multi-session data would make blocked CV less brittle.
- **Only one model studied in depth.** We picked KNN because it is
  the most informative choice for studying the leakage problem. The
  alternative-classifier appendix is a sanity check, not a primary
  result.
- **Per-sample voltages only.** No windowed features (rolling means,
  variances, lagged values). A windowed representation would give
  KNN something more meaningful to nearest-neighbour on, and would
  likely close some of the leakage gap. We deliberately kept features
  per-sample so the leaky-vs-honest comparison is apples-to-apples.
- **Future work.** Replace per-sample voltages with windowed
  features; test cross-subject generalisation on a multi-subject
  corpus; extend the `k` sweep to characterise the leakage curve more
  precisely.

---

## 7. Conclusion

On the UCI EEG Eye State dataset, using only the methods covered in
COGS 109 SP26, we picked classification as our
approach, KNN as our model, and a `k`-sweep over a log-spaced grid as
our model-selection procedure. We ran that same procedure under
three different cross-validation schemes (shuffled, naive blocked,
stratified blocked) and found that all three schemes pick `k = 1` as
optimal, but report dramatically different picked accuracies: 97.3%
± 0.4% (shuffled), 50.0% ± 10.7% (naive blocked), and 77.8% ± 2.7%
(stratified blocked). A small sanity-check appendix on three
alternative classifiers (LDA, PCA→LDA, PCR) showed leakage gaps of
only 2–6 pp, much smaller than KNN's 19.5 pp gap, exactly as theory
predicts for a classifier that exploits per-sample similarity. The
project is a worked example of how a methodologically sound model
selection procedure can still be misled by an upstream choice — the
CV scheme — that has nothing to do with the model itself.

> **Shuffled CV says we have a 97% classifier. Honest stratified blocked CV says we have a 78% classifier. The difference is leakage, not signal.**

---

## References

1. Roesler, O. (2013). *EEG Eye State Data Set*. UCI Machine Learning
   Repository, id #264. <https://archive.ics.uci.edu/dataset/264/eeg+eye+state>
2. James, G., Witten, D., Hastie, T., Tibshirani, R. (2021). *An
   Introduction to Statistical Learning, with Applications in R*,
   2nd ed. Springer.
3. Bergmeir, C., Benítez, J. M. (2012). On the use of cross-validation
   for time series predictor evaluation. *Information Sciences*
   191:192–213.
4. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion,
   B., et al. (2011). Scikit-learn: Machine learning in Python.
   *Journal of Machine Learning Research* 12:2825–2830. (Used here
   for the `KNeighborsClassifier`, `LinearDiscriminantAnalysis`,
   `PCA`, and `LinearRegression` implementations.)
