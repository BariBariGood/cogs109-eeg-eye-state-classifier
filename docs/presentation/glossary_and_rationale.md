# COGS 109 Project — Glossary, Rationale, and Implementation Walkthrough

**Project:** Classification of EEG Eye-State via KNN — A Cross-Validation Honesty Study
**Audience:** This is a *learning* document, not a script. Read it cover-to-cover so every term on the poster and every design choice in the project has both a plain-English **what** and an honest **why** in your head.

---

## How to use this doc
- **Section 1 — Glossary**: every complex word on the poster, in plain English. If a grader uses a term you blank on, search here.
- **Section 2 — Every decision + why we made it**: a paired "what we did" / "why we did it" for every methodological choice. Read this section twice. The "why" is what graders are testing.
- **Section 3 — High-level implementation**: how the code is structured, what each file does, what libraries we used. So you can answer *"walk me through your codebase"* without panicking.
- **Section 4 — End-to-end pipeline walkthrough**: what happens when you run `make all` from `eeg_eye_state.csv` raw all the way to the final poster.

---

# Section 1 — Glossary

Every term in alphabetical order. If you don't know one, learn it before the symposium.

### Accuracy
The fraction of samples the classifier got right. (True positives + true negatives) / total samples. We use it as our headline metric throughout. Not the same as precision or recall — accuracy treats both classes equally.

### Argmax
"The argument that maximises". `argmax_k accuracy(k)` means "the value of `k` that gives the largest accuracy". We use argmax-of-CV-accuracy as our model-selection rule: among all the `k` values we swept, pick the one with the best mean CV accuracy.

### Autocorrelation
A statistical measure of how similar a signal is to a delayed copy of itself. **Lag-1 autocorrelation** is the correlation between sample `t` and sample `t+1`. Our EEG channels have lag-1 autocorrelation ≈ 0.997, which means consecutive samples are almost numerically identical — they share 99.7% of their variance.

### Baseline (majority-class baseline)
The accuracy a "dumb" classifier achieves by always predicting the most common class. On our data, 55.12% of samples are eyes-open, so a classifier that always predicts "eyes open" gets 55.12% accuracy. Any honest classifier needs to beat that — and our headline 77.8% beats it by about 23 percentage points.

### Bias-variance tradeoff
A core machine-learning concept. **Bias** is error from a too-simple model (oversmoothing, missing real patterns). **Variance** is error from a too-flexible model (overfitting, memorising noise). Small `k` in KNN is low-bias / high-variance; large `k` is high-bias / low-variance. We sweep `k` precisely to find the right tradeoff on our data.

### Binary classification
A classification problem with exactly two classes. Our label is binary: 0 = eyes open, 1 = eyes closed.

### Blocked CV (blocked cross-validation)
A cross-validation scheme where folds are contiguous chunks of the time series, not random shuffles. Designed for time-series data so that adjacent samples don't end up in different folds (which would leak temporal autocorrelation).

### Channel (EEG channel)
A single voltage signal from one electrode on the EEG headset. Our dataset has 14 channels: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4. The letter codes refer to anatomical positions on the scalp (F=frontal, T=temporal, P=parietal, O=occipital).

### Classification
A type of supervised learning where the goal is to predict a categorical label. Contrast with regression, which predicts a continuous value. Our project does binary classification (eyes open vs. closed).

### Confusion matrix
A 2×2 table for binary classification. Rows = true labels, columns = predicted labels. Cells: true positives (TP), false positives (FP), true negatives (TN), false negatives (FN). From it we derive sensitivity and specificity. We show confusion matrices for the picked KNN model in figure 15.

### Cross-validation (CV)
A technique to estimate how well a model generalises to unseen data, using only training data. Standard recipe: split the training set into `K` equal folds, train on `K-1` of them, test on the held-out one, repeat `K` times rotating which fold is held out, average the per-fold accuracies. We use 5-fold CV throughout.

### Decision boundary
The surface in feature space where the classifier flips from predicting one class to predicting the other. For KNN with `k=1` it's the *Voronoi diagram* of the training points; for KNN with large `k` it's smoother.

### Deterministic
A computation that always gives the same result for the same inputs. Our CV fold-index generators are deterministic given a random seed, so anyone re-running our code gets the exact same folds. Important for reproducibility.

### Dimensionality reduction
Reducing the number of features in your data while keeping as much information as possible. PCA is the standard tool. We use PCA inside the PCA→LDA and PCR sanity checks, and informally for EDA (figure 09 PCA scatter).

### EEG (electroencephalography)
A non-invasive technique for measuring brain electrical activity from the scalp via electrodes. The dataset here uses an Emotiv EPOC headset — a 14-channel consumer-grade EEG device.

### Emotiv EPOC
The specific consumer-grade EEG headset used to collect this dataset. Has 14 electrodes at fixed scalp positions, samples at ~128 Hz, and is much cheaper than research-grade EEG (which is why the data is noisier).

### Euclidean distance
The standard "straight-line" distance between two points in feature space. For two 14-D points `a` and `b`: `sqrt(sum((a_i - b_i)^2 for i in 1..14))`. KNN uses Euclidean distance to find neighbours.

### Feature space
The multidimensional space whose axes are the features of your data. Our feature space is 14-dimensional — one dimension per EEG channel. Every sample is a point in this 14-D space, and KNN finds neighbours by Euclidean distance in this space.

### Fold (CV fold)
One of the `K` equal-sized subsets that cross-validation splits the data into. In 5-fold CV there are 5 folds; each fold takes a turn being the held-out test set.

### Generalisation
A model's ability to perform well on data it has never seen during training. This is the actual goal of machine learning — high training accuracy alone doesn't matter if the model fails on new data. The whole CV-scheme story we tell is about how shuffled CV *overestimates* generalisation on autocorrelated data.

### Holdout (holdout test partition)
A subset of data that the model never touches during training or validation. We use a chronological 80/20 split: the first 80% of samples are the development partition (used for training + CV), and the last 20% is the held-out test partition. We do *not* over-interpret our holdout accuracy because the last 20% of this particular recording happens to be 91% class-0 (the subject's eyes were mostly open at the end).

### Honest (honest CV / honest evaluator)
A CV scheme that doesn't leak information between train and test folds. **In our context, "honest" specifically means our stratified blocked CV** — temporal autocorrelation is respected (adjacent samples stay in the same fold) *and* fold-level class balance is preserved. The opposite is "leaky", which we use to describe shuffled CV.

### Hyperparameter
A parameter of a model that you set *before* training, as opposed to one the training procedure learns from data. For KNN the only hyperparameter is `k`. We pick `k` via cross-validation in our model-selection procedure.

### i.i.d. (independent and identically distributed)
A standard statistical assumption that each sample is drawn independently from the same distribution. Shuffled CV implicitly assumes the data is i.i.d. Our EEG data is *not* i.i.d. — adjacent samples are highly dependent (lag-1 r = 0.997) — which is exactly why shuffled CV fails on it.

### k (in KNN)
The hyperparameter of KNN — how many nearest neighbours to consider when classifying a new point. `k=1` means "just use the single closest training point". `k=21` means "take a majority vote among the 21 closest training points". We sweep `k` over a log-spaced odd-only grid from 1 to 201.

### K-means
An unsupervised clustering algorithm that partitions data into `K` clusters by iteratively assigning points to the nearest cluster centroid and updating centroids. We ran K-means with `k=2` on the channels as an exploratory check — the clusters did *not* line up with the eye-state label, confirming that the eye-state distinction needs supervised learning to recover.

### k-fold
The cross-validation scheme where you split data into `K` equal folds. We use `K=5` throughout — five folds, each containing ~20% of the development partition.

### KNN (K-Nearest Neighbours)
A simple non-parametric classifier. To classify a new point, find the `k` training points closest to it (Euclidean distance in feature space), take a majority vote of their labels. No training in the usual sense — you just store all training points. The hyperparameter is `k`.

### Label
The supervised target variable that the classifier learns to predict. Our label is `eyeDetection`, a binary indicator of whether the subject's eyes were open (0) or closed (1) at each sample.

### Lag
The number of samples between two points in time. "Lag-1" means 1 sample apart, "lag-100" means 100 samples apart. At 128 Hz, lag-1 is ~7.8 ms and lag-100 is ~0.78 s.

### LDA (Linear Discriminant Analysis)
A linear classifier that projects multi-dimensional features onto the 1-D axis that best separates the classes by maximising between-class variance and minimising within-class variance. We use it as a sanity-check alternative classifier. Its leakage gap is only 5.6 pp — much smaller than KNN's 19.5 pp — which supports our hypothesis that KNN's gap is per-sample-similarity-specific.

### Leakage (data leakage)
Any situation where information from the test set indirectly influences training. **In our project, "leakage" specifically refers to lag-1 autocorrelation leakage** — under shuffled CV, the temporal neighbour of every test sample lands in the training fold and carries nearly the same label, so the classifier effectively "looks up" the answer rather than generalising.

### Leakage gap
**A term we coined for this project.** The difference between an estimator's leaky shuffled-CV accuracy and its honest stratified blocked-CV accuracy at the *same picked hyperparameter*. For KNN this is 0.973 − 0.778 = 0.195 = **19.5 percentage points** — the project's headline number.

### Leaky (leaky CV)
A CV scheme that allows information from the test fold to "leak" into the training fold. We use "leaky" to describe shuffled CV on autocorrelated time series — adjacent samples are nearly identical, so even though they're in different folds, the train fold contains near-duplicates of test-fold samples.

### Log-spaced
A grid of values where the *ratios* between adjacent values are roughly constant, not the differences. Example: `{1, 3, 7, 15, 31, ...}` — each value is roughly double the previous. Useful for hyperparameters where the interesting range spans many orders of magnitude. We use it for the `k` grid because the small-`k` region matters most.

### Mean accuracy
The arithmetic average of per-fold accuracies across the 5 CV folds. Our headline numbers (97.3%, 50.0%, 77.8%) are all mean accuracies.

### Model selection
The procedure of picking the best version of a model. In our project: sweep `k` over a grid, compute 5-fold CV accuracy for each, pick the `k` with the highest mean accuracy. The hyperparameter `k` is what's being *selected*.

### Naive blocked CV
The 5-contiguous-chunks CV scheme without any class-stratification correction. We use "naive" because it has an obvious failure mode — fold class imbalance — that stratified blocked CV fixes.

### Non-parametric
A model class that does not assume a fixed mathematical form for the relationship between features and label. KNN is non-parametric — it doesn't fit a line, plane, or curve; it just stores training points and looks them up at prediction time.

### Outlier
A data point that's far from the rest. We flag outliers as samples that exceed 4 standard deviations from the channel mean on any of the 14 channels, and drop them — 4 out of 14,980 samples meet this criterion (they're voltage-saturation artefacts where an electrode briefly disconnected).

### Overfitting
When a model fits the training data too closely, including noise, and consequently fails on new data. Small-`k` KNN can overfit, but on this dataset `k=1` is the optimum across all three CV schemes — overfitting isn't the limiting factor; CV-scheme-induced overestimation is.

### PCA (Principal Component Analysis)
A dimensionality-reduction technique that finds the orthogonal directions of maximum variance in the data. We use PCA inside the PCA→LDA and PCR baselines (sweep the number of retained components), and informally for EDA in figure 09.

### PCA → LDA
A sanity-check alternative classifier: first project the 14-D features onto `n` principal components, then run LDA on the projected features. We sweep `n` over `{1, 2, 3, 5, 7, 10}` and pick `n=3` as best. Its leakage gap is 2.4 pp — supports our KNN-vulnerability hypothesis.

### PCR (Principal Component Regression)
Linear regression on the first `n` principal components. We adapt it as a classifier by thresholding the regression output at 0.5. We sweep `n` over the same grid as PCA→LDA. Its leakage gap is 2.8 pp.

### Per-sample similarity
The property that a model's prediction depends on the identity of one or a few specific training samples (rather than aggregated statistics over many). KNN with `k=1` is the extreme case — the prediction is literally the label of the single nearest training sample. This is why KNN is uniquely vulnerable to lag-1 leakage.

### Picked accuracy
The mean CV accuracy at the picked hyperparameter. So if the procedure picks `k=1` and the mean accuracy at `k=1` is 0.778, the picked accuracy is 0.778.

### Picked k (picked hyperparameter)
The value of `k` (or whatever hyperparameter) selected by argmax-of-CV-accuracy. All three of our CV schemes pick `k=1`, which is itself a notable result — the model-selection procedure is robust even when the *accuracy it reports* isn't.

### Preprocessing
Data cleaning and transformation done before training. Our pipeline does: drop outliers (4 SD on any channel), chronological 80/20 split with a 64-sample seam gap, z-score each channel using training-partition statistics only.

### Reproducibility
The property that someone else can re-run your code and get the exact same numbers. We achieve this through deterministic RNG seeds (`seed=42` everywhere it matters), pinned dependency versions in `requirements.txt`, and a `make all` target that regenerates everything from raw data.

### Robust (model-selection robustness)
A property of our model-selection procedure: even when the CV scheme changes dramatically (shuffled vs. blocked vs. stratified blocked), the *picked* `k` doesn't change — it's `k=1` under all three. That's the procedure being robust. What changes is the *picked accuracy*.

### Sample
One row of the dataset — 14 channel values + one label value at one moment in time. Sometimes called a "data point" or "observation". We have 14,980 samples total.

### Sampling rate
How many samples per second the EEG recorded. Ours is ~128 Hz — 128 samples per second — so consecutive samples are ~7.8 ms apart.

### Seam gap
A buffer of samples between the train and test partitions that belong to neither. We use a 64-sample seam gap (half a second at 128 Hz) between the development and test partitions to prevent boundary contamination — making sure samples at the edge of the train partition don't share information with samples at the edge of the test partition.

### Seed (random seed)
An integer that initialises a random number generator. We pin all seeds (`seed=42`) so every CV fold assignment is deterministic and reproducible.

### Segment (in stratified blocked CV)
One of the 100 short contiguous time chunks we cut the training partition into. Each segment is ~120 samples (≈0.9 s at 128 Hz). Segments are the atomic unit of stratified blocked CV — we assign whole segments to folds, never split a segment across folds.

### Sensitivity (recall, true positive rate)
TP / (TP + FN) — of all the actual positives (eyes-closed samples), what fraction did the classifier correctly catch? Reported in the confusion matrices.

### Shuffled CV (i.i.d. CV, random k-fold)
The standard cross-validation scheme: assign samples to folds uniformly at random. The default for most students of this dataset. **Our leaky baseline** — on an autocorrelated time series, shuffling lets adjacent samples land in different folds, which leaks the lag-1 signal.

### Single-subject (study)
Data collected from only one person. Ours is single-subject — all 14,980 samples come from the same person, in a single ~117-second recording. This is a limitation: our results don't tell us how the model would generalise to a *different* person.

### Specificity (true negative rate)
TN / (TN + FP) — of all the actual negatives (eyes-open samples), what fraction did the classifier correctly reject? Reported in the confusion matrices.

### Standardisation (z-scoring)
A preprocessing step: for each feature, subtract the mean and divide by the standard deviation. Result: every feature has mean 0 and std 1. Important for KNN because Euclidean distance is scale-sensitive — without it, a 100-µV channel would dominate over a 10-µV channel. We use training-partition statistics only (no test leakage).

### Stratification
A procedure for assigning samples to subsets (folds, train/test) such that each subset has roughly the same class distribution as the whole dataset. Critical when classes are imbalanced. We use stratification at the segment level in stratified blocked CV.

### Stratified blocked CV
**The CV scheme we recommend as the honest evaluator.** Cuts the training partition into 100 short contiguous segments, then assigns segments to folds in a class-stratified round-robin so each fold has roughly the same class balance as the whole training set. Combines the temporal-isolation guarantees of blocked CV with the class-balance guarantees of stratified CV.

### Supervised learning
Machine learning with labelled training data — the model learns to predict labels from features. Contrast with unsupervised learning (no labels — e.g. clustering, dimensionality reduction). Classification is a flavour of supervised learning.

### Take-home message
The short, blunt summary of what your project shows. Ours: *"Shuffled CV says we have a 97% classifier. Honest stratified blocked CV says we have a 78% classifier. The difference is leakage, not signal."*

### Test set / test partition
Data that the model never sees during training or hyperparameter selection. Our chronological 80/20 split holds out the last 20% of samples as the final test partition.

### Time series
A sequence of measurements ordered in time. Our EEG recording is a time series — each sample has a specific time stamp (implicit in row order). Time-series data violates the i.i.d. assumption that shuffled CV implicitly relies on.

### Training partition / development partition
The data the model trains on (and that CV is run inside of). Our development partition is the first 80% of samples after outlier removal — 11,917 samples.

### z-score
See *Standardisation*.

---

# Section 2 — Every decision + why we made it

For every "what we did" choice in the project, here's the paired "why we did it."

### Why pick THIS dataset (UCI #264)?
**What:** UCI ML Repository dataset #264 — Roesler's EEG Eye State recording.
**Why:** Three reasons. (1) It's a clean, small-scale, single-task EEG dataset — no confounds from multiple subjects or sessions. (2) It's heavily tutorialised online — dozens of student write-ups and Kaggle notebooks report 95-98% accuracy on it using shuffled CV, which gave us a clear "received wisdom" to interrogate. (3) Its time-series structure (lag-1 r = 0.997) makes the CV-scheme question visible — if you tried this story on i.i.d. tabular data, there'd be no story to tell.

### Why pick CLASSIFICATION as the data-analysis approach?
**What:** Classification, not regression / clustering / dimensionality reduction.
**Why:** The label (`eyeDetection`) is **binary categorical** — 0 or 1, with no natural ordering or magnitude. Regression would force us to interpret 0.7 as "70% eyes closed", which has no physical meaning. Clustering would discard the label entirely and ask whether eye-state structure emerges *unsupervised* — we ran K-means with `k=2` as a side check and the clusters did *not* line up with the label, which actually confirmed that the eye-state distinction needs supervised learning. Classification was the obvious approach choice.

### Why pick KNN as the model?
**What:** K-Nearest Neighbours with Euclidean distance in 14-D z-scored channel space.
**Why:** Three reasons. (1) KNN is a non-parametric classifier in the COGS 109 palette. (2) KNN has *exactly one* tunable hyperparameter (`k`), which makes the model-selection narrative crystal clean — there's no ambiguity about what's being swept. (3) KNN's decision rule is *per-sample similarity* — it predicts based on the identity of a few specific training points. That makes it the model most exposed to lag-1 autocorrelation leakage, which was the methodological question we wanted to study. The other COGS 109 classifiers (LDA, PCA→LDA, PCR) all average over many samples before classifying — they smear out the per-sample similarity that drives leakage. KNN was therefore the cleanest test case.

### Why sweep `k` over a LOG-SPACED grid?
**What:** Grid is `{1, 3, 5, 7, 11, 15, 21, 31, 51, 75, 99, 151, 201}` — log-spaced.
**Why:** The interesting region for KNN is small `k` (where overfitting / leakage is strongest). A linear grid like `{1, 2, 3, ..., 200}` would waste 90% of its evaluations in the large-`k` region where the curve is nearly flat. Log-spacing doubles the resolution at small `k` for free.

### Why ODD-ONLY `k` values?
**What:** `{1, 3, 5, 7, 11, 15, 21, ...}` — no even values.
**Why:** With a binary label and `k=2` you can have a 1-1 tie among the neighbours, with no clean winner. The standard fix is to restrict `k` to odd numbers, which guarantees a majority. (Some implementations break ties randomly, but odd-only is the cleaner choice.)

### Why pick `k` by ARGMAX of CV accuracy?
**What:** The model-selection rule is "choose the `k` with the highest mean 5-fold CV accuracy."
**Why:** This is the standard model-selection procedure for any classifier with a single tunable hyperparameter. The rule is identical across all three CV schemes — only the fold-assignment rule changes — which is what makes the three-way comparison apples-to-apples.

### Why 5-FOLD CV?
**What:** `K = 5` folds.
**Why:** 5-fold is the standard bias-variance tradeoff for CV. With ~12,000 training samples, each fold has ~2,400 samples — plenty to get a stable per-fold accuracy. Going to 10-fold would double our compute without meaningfully changing the conclusions; going to 3-fold would give noisier per-fold estimates. 5 is the textbook default.

### Why SHUFFLED CV in the comparison at all?
**What:** We include shuffled CV as the leaky baseline.
**Why:** Because it's what most students of this dataset *actually* report — it's the leakiest possible scheme and also the most common default (sklearn's `KFold(shuffle=True)`). Including it makes the methodological point land: the numbers people are quoting on this dataset are inflated by leakage, and we can quantify by how much.

### Why NAIVE BLOCKED CV in the comparison?
**What:** Five contiguous time chunks, no shuffling.
**Why:** It's the obvious "fix" for the leakage problem — and it's important to show that the obvious fix doesn't work. Naive blocked CV eliminates the lag-1 leakage but introduces a new problem: per-fold class imbalance. The label runs in our recording are unevenly distributed, so individual folds can end up at 20% class-0 vs. the global ~54%. Fold accuracy collapses to ~50%, below the majority-class baseline. We include this scheme to make the failure mode visible — it sets up *why* stratified blocked CV is necessary.

### Why STRATIFIED BLOCKED CV as the honest evaluator?
**What:** Cut the training partition into 100 short contiguous segments, then assign segments to 5 folds in a class-stratified round-robin.
**Why:** It combines the strengths of blocked CV (temporal isolation — adjacent samples never land in different folds at the segment boundary) and stratified CV (class balance preserved across folds). Segments are short enough (~0.9 s) that the per-segment label is nearly always constant, but long enough that most segments don't straddle a label transition. This is the cleanest honest evaluator we could build using only study-guide-compliant methods.

### Why 100 SEGMENTS specifically?
**What:** 100 segments × ~120 samples each.
**Why:** Two design constraints. **Lower bound on segment count**: with only 5 folds and 1 segment per fold, we'd have just 5 contiguous chunks — that's naive blocked CV, which fails. We need many segments per fold so class-stratified assignment is possible. **Upper bound on segment count**: with too many segments (say, 1000), each segment is only ~12 samples long, almost certainly within a label run, and stratification becomes the only thing distinguishing this from shuffled CV — we lose the temporal-locality benefit. 100 is a clean compromise: ~120-sample segments are *long* enough that most don't straddle label transitions, *short* enough that 5-fold stratified assignment is meaningful.

### Why a CHRONOLOGICAL 80/20 SPLIT for the holdout?
**What:** First 80% of samples = development partition, last 20% = held-out test partition.
**Why:** Chronological splits simulate deployment — you train on the past and predict the future. The alternative (random 80/20) would have the same leakage problem as shuffled CV. The downside is that the last 20% of *this particular* recording happens to be 91% class-0 (the subject's eyes were mostly open at the end), so we don't over-interpret holdout accuracy in isolation — the per-fold CV numbers are the more meaningful summary.

### Why a 64-SAMPLE SEAM GAP between train and test?
**What:** 64 samples (half a second at 128 Hz) immediately before the test partition are dropped — they belong to neither train nor test.
**Why:** To prevent boundary contamination. At lag-1 autocorrelation of 0.997, a sample at position `i` is nearly identical to the sample at `i-1`, `i-2`, etc. If the last sample in the train partition is at position `t` and the first sample in the test partition is at position `t+1`, the very last train sample is leaked into the very first test prediction. A 64-sample buffer (~0.5 s) is more than long enough to ensure that the lag-1 autocorrelation window of the train partition has fully decayed before the test partition starts.

### Why DROP OUTLIERS at 4 SD?
**What:** Samples that exceed 4 standard deviations from the channel mean on any of 14 channels are dropped. This removes 4 samples out of 14,980.
**Why:** Conservative artefact removal — 4 SD is a strict threshold. The 4 dropped samples are voltage-saturation artefacts (one or more electrodes briefly disconnected). We kept the threshold strict because the dataset is small and we wanted to confirm the outliers were artefacts, not signal. We verified that the picked accuracy is essentially identical at 4 SD vs. 10 SD thresholds because both drop the same 4 samples.

### Why Z-SCORE on TRAINING STATS ONLY?
**What:** Compute channel means and stds on the training partition; apply that transform to train, validation, and test.
**Why:** Z-scoring with test statistics leaks information from the test set into preprocessing. If we computed the mean over the whole dataset and applied it everywhere, the training data would be implicitly "told" what the test distribution looks like — that's a form of data leakage, even before any model touches the data. The standard fix: estimate transform parameters on train only.

### Why use EUCLIDEAN DISTANCE for KNN?
**What:** Distance metric for nearest-neighbour search is Euclidean (L2 norm).
**Why:** It's the standard distance metric for KNN on standardised features — the textbook default. Other metrics (Manhattan, Mahalanobis, cosine) are possible but wouldn't materially change the leakage story, because the leakage comes from lag-1 sample *similarity*, which is high under any reasonable metric on standardised channels. We didn't experiment with alternative metrics because they'd be a different research question.

### Why include LDA / PCA→LDA / PCR as alternative classifiers?
**What:** Three sanity-check classifiers, demoted to a "supporting evidence" panel.
**Why:** To rule out the possibility that the leakage gap is a generic property of *this dataset* rather than a model-specific property of KNN. Theory predicts: KNN's gap should be large (it relies on per-sample similarity); LDA / PCA→LDA / PCR's gaps should be smaller (they average over many samples before classifying). We measured: KNN 19.5 pp vs. LDA 5.6 pp vs. PCA→LDA 2.4 pp vs. PCR 2.8 pp — exactly the predicted pattern. This makes the KNN-specific story much stronger.

### Why NOT logistic regression, SVM, trees, neural nets, FFT features?
**What:** We don't use any model class or feature type outside the COGS 109 SP26 study-guide palette.
**Why:** The assignment scope is explicit — only methods from the study guide. Logistic regression, SVMs, trees, and neural nets are out. Spectral / FFT features (which are standard in real-world BCI work) are also out. Two consequences: (1) we can't claim a "best EEG classifier" result — we can only claim a methodological result within the study-guide palette; (2) the methodological result is *still meaningful* because the leakage problem we documented is independent of model complexity — a deep net under shuffled CV would show the same inflation.

### Why pick `k = 1` as the final model?
**What:** The headline KNN model is `k = 1`.
**Why:** We didn't *pick* it directly — the model-selection procedure picked it. Argmax-of-CV-accuracy across the log-spaced grid landed on `k=1` under *all three* CV schemes, which is itself a notable robustness result. The reason `k=1` wins: the label is piecewise-constant over very long runs (only 23 transitions in 14,980 samples), so even under blocked CV, the nearest training sample in feature space often comes from a region where the label hasn't changed yet — copying its label is a good bet. Larger `k` introduces averaging that doesn't help here because the local label is already homogeneous.

### Why is the FINAL ACCURACY 77.8% and not 97.3%?
**What:** The headline accuracy we report is 77.8% (stratified blocked CV), not 97.3% (shuffled CV).
**Why:** Because the 97.3% is inflated by lag-1 autocorrelation leakage. Under shuffled CV, the temporal neighbour of every test sample is in the training fold with ~80% probability and carries the same label with ~99.7% probability — the model is essentially being asked a question it can answer by lookup. Under stratified blocked CV, the temporal neighbour is reserved into the same segment as the test sample (never in the training fold), so the model has to actually generalise. 77.8% reflects what the model has actually learned; 97.3% reflects what leakage makes it look like it learned.

### Why is 77.8% still a meaningful result?
**What:** We claim 77.8% is a meaningful accuracy, not a failure.
**Why:** Because the majority-class baseline is 55.12%, so 77.8% beats baseline by ~23 percentage points. The classifier is genuinely learning *something* about the EEG-to-eye-state mapping — it's just much less impressive than the shuffled-CV number implies. For a real-world BCI application, 77.8% within-subject accuracy at temporally-isolated evaluation is a plausible deployment estimate; 97.3% is fantasy.

---

# Section 3 — High-level implementation

How the code is structured. If a grader asks "walk me through your codebase", here's what you say.

### Languages and libraries
- **Python 3.12** throughout.
- **numpy / pandas** for array and tabular data.
- **scikit-learn** for the standard ML primitives we use: `KNeighborsClassifier`, `LinearDiscriminantAnalysis`, `PCA`, `LinearRegression` (used for PCR), `KMeans`.
- **matplotlib** for every figure.
- **python-pptx** for generating the poster `.pptx` files programmatically.
- **pytest** for the test suite (18 tests covering data loading, preprocessing, CV folding, modelling, poster generation).

### Repo layout

```
src/                   # Reusable Python modules
├── data.py            # Load + clean raw CSV; build train/test partitions
├── cv.py              # The three CV fold-index generators (the methodological core)
├── models.py          # Sklearn wrappers around KNN, LDA, PCA→LDA, PCR
├── evaluate.py        # cv_score() — runs N folds, returns mean ± std
└── plotting.py        # Shared matplotlib styling for figures

scripts/               # One-off, command-line driven entry points
├── preprocess.py      # Reads raw CSV → writes processed train/test CSVs + a manifest
├── build_poster.py    # Reads figures/tables → emits poster.pptx (main Polished Classic variant)
├── regenerate_figure_11.py   # Standalone script to regenerate the headline k-sweep figure
└── variants/          # Per-design-variant pptx builders (build_editorial.py, build_academic.py, etc.)

notebooks/             # Reproducible analysis notebooks
├── 00_fetch_data.ipynb     # Downloads UCI #264, writes raw CSV
├── 01_eda.ipynb            # Exploratory data analysis — figures 01-09
└── 02_modeling.ipynb       # KNN k-sweep + alternative classifiers — figures 10-15 + tables 01-04

tests/                 # pytest test suite
├── test_smoke.py            # Data loading + preprocessing + CV fold determinism
├── test_modeling.py         # Per-model fit/predict shape + CV scoring determinism
└── test_poster.py           # Poster builder produces a valid pptx file

figures/               # 15 PNG figures, all regenerable from notebooks
tables/                # 4 CSV tables of per-model and per-scheme accuracy
poster/                # The poster.pptx + 4 design variants
report/final_report.md # Paper-style writeup
README.md              # Project quickstart
Makefile               # make all / make poster / make variants targets
```

### The three CV functions (the methodological core)

All three live in `src/cv.py` and have a deterministic signature: given `n_samples` (or `labels` for the stratified version), `n_splits`, and `seed`, return a list of `(train_indices, test_indices)` tuples — one per fold. The notebook then loops over those tuples, training and evaluating once per fold.

- **`shuffled_kfold_indices(n_samples, n_splits=5, seed=42)`** — Uniform-random shuffle of `range(n_samples)`, then slice into 5 equal contiguous chunks of the shuffled array. Each chunk's indices are the test fold; the complement is the train fold.

- **`blocked_kfold_indices(n_samples, n_splits=5)`** — No shuffling at all. Slice `range(n_samples)` directly into 5 contiguous chunks. Each chunk is one test fold.

- **`stratified_blocked_kfold_indices(labels, n_splits=5, n_segments=50, seed=42)`** — Cut `range(n_samples)` into `n_segments` contiguous segments. For each segment, compute its class-1 proportion. Sort segments by that proportion. Then walk down the sorted list and assign segments to folds in chunks of `n_splits`, with the assignment within each chunk being a deterministic random permutation seeded by `seed`. This guarantees each fold gets one segment from each n-tile of the sorted-by-class-balance list, so every fold ends up class-balanced. (For the headline experiment we use `n_segments=100`.)

### The k-sweep pipeline (notebook 02_modeling)

Roughly this in pseudocode:

```python
from src.cv import (
    shuffled_kfold_indices,
    blocked_kfold_indices,
    stratified_blocked_kfold_indices,
)
from src.evaluate import cv_score
from sklearn.neighbors import KNeighborsClassifier

K_GRID = [1, 3, 5, 7, 11, 15, 21, 31, 51, 75, 99, 151, 201]
SCHEMES = {
    "shuffled":           shuffled_kfold_indices(n_samples),
    "naive_blocked":      blocked_kfold_indices(n_samples),
    "stratified_blocked": stratified_blocked_kfold_indices(labels, n_segments=100),
}

results = {}
for scheme_name, folds in SCHEMES.items():
    for k in K_GRID:
        model = KNeighborsClassifier(n_neighbors=k)
        mean_acc, std_acc = cv_score(model, X_train, y_train, folds)
        results[(scheme_name, k)] = (mean_acc, std_acc)

# argmax-of-mean-accuracy per scheme
for scheme_name in SCHEMES:
    picked_k = max(K_GRID, key=lambda k: results[(scheme_name, k)][0])
    print(scheme_name, picked_k, results[(scheme_name, picked_k)])
```

That's the entire experiment in 30 lines. Same picker rule across all three schemes; only `folds` changes.

### How the poster is generated (`scripts/build_poster.py`)

`python-pptx` lets you build a `.pptx` file in code: create slide → add text boxes with content + styling → add image boxes pointing at PNG files in `figures/` → save. Our script encodes the 11-panel layout as a list of dictionaries, each entry specifying position, size, text content, and font styling. The four design variants (`scripts/variants/build_editorial.py`, etc.) reuse a shared layout primitive in `scripts/variants/_common.py` and override colours, fonts, and panel positioning.

This means the entire poster is reproducible from code — no manual edits, no PowerPoint clicks. If a number on the poster changes, you just edit the source and re-run `make poster`.

### How to reproduce everything from scratch

```bash
git clone https://github.com/BariBariGood/cogs109-eeg-eye-state-classifier
pip install -r requirements.txt
make all       # preprocess + EDA + modeling, regenerates all figures + tables
make poster    # builds poster/poster.pptx (Polished Classic)
make variants  # builds the other 3 design variants in poster/variants/
pytest tests/  # 18 tests, all should pass
```

End-to-end runtime: ~30-60 seconds on a modern laptop. The dataset is small enough that nothing needs a GPU.

---

# Section 4 — End-to-end pipeline walkthrough

What actually happens when you run `make all`. If someone asks "how does your code go from a raw CSV to your headline number?", here's the linear story.

### Step 1 — Fetch
**File:** `notebooks/00_fetch_data.ipynb`
- Downloads UCI ML Repository dataset #264 (14,980 samples × 15 columns: 14 EEG channels + 1 label).
- Writes `data/raw/eeg_eye_state.csv` + a `manifest.json` capturing source URL and SHA256.

### Step 2 — Preprocess
**File:** `scripts/preprocess.py`
- Loads the raw CSV.
- Computes per-channel mean and std on the full dataset (for outlier detection only).
- Flags samples that exceed 4 SD on any channel — drops 4 samples → 14,976 remain.
- Does a chronological 80/20 split with a 64-sample seam gap: development partition = first 11,917 samples, test partition = last 2,995 samples.
- Re-computes per-channel mean and std *on the development partition only* (this time for z-scoring).
- Applies that transform to both development and test partitions.
- Writes `data/processed/eeg_train.csv` and `data/processed/eeg_test.csv`.

### Step 3 — EDA
**File:** `notebooks/01_eda.ipynb`
- Loads the processed development partition.
- Generates figures 01-09: class balance, channel boxplots, channel histograms, time-series with label, channel correlation heatmap, channel dendrogram (hierarchical clustering on the channel correlation matrix), K-means cluster vs. label, label autocorrelation, PCA scatter.

### Step 4 — Modelling (the headline experiment)
**File:** `notebooks/02_modeling.ipynb`
- Builds CV folds for all three schemes using `src/cv.py`.
- Sweeps `k ∈ {1, 3, 5, ..., 201}` for KNN under each scheme, computes per-fold accuracy via `src/evaluate.py:cv_score`, aggregates as mean ± std.
- Repeats the same idea for the three alternative classifiers (LDA, PCA→LDA, PCR-as-classifier).
- Picks per-scheme argmax-of-mean-accuracy `k` (or `n_components`).
- Saves figures 10-15 (LDA coefficients, k-sweep, components sweeps, headline three-way comparison, confusion matrices) and tables 01-04 (per-model per-scheme accuracy + the k-sweep CSV backing figure 11).

### Step 5 — Poster
**File:** `scripts/build_poster.py` and `scripts/variants/build_all.py`
- Reads the figures and tables.
- Lays out the 11-panel poster as a list of python-pptx text and image boxes.
- Writes `poster/poster.pptx` + `poster/poster_preview.png`.
- Variants script generates the 3 alternative design variants.

### Step 6 — Tests
**Directory:** `tests/`
- 18 tests covering: data loads correctly, outlier drop removes the right 4 samples, CV folds are deterministic, blocked folds are contiguous, stratified blocked folds cover every sample exactly once and balance class per fold, the poster builder produces a valid pptx file.
- All 18 pass deterministically.

### What this whole pipeline is verifying
> Same model. Same data. Same `k`. Different CV schemes. Three completely different "accuracies". The 19.5-percentage-point gap between the leaky shuffled-CV number and the honest stratified blocked-CV number is leakage, not signal — and the whole pipeline above is the auditable evidence that supports that claim.

---

**End of glossary + rationale + implementation walkthrough.** If you've read this cover-to-cover, every term, every choice, and every line of code on the poster should be fully justified in your head.
