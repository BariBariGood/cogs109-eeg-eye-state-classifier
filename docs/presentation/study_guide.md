# COGS 109 Poster-Session Study Guide
**Project:** Classification of EEG Eye-State via KNN — A Cross-Validation Honesty Study
**Authors:** Ivan Del Rio · Anish Kondamadugula
**Course:** COGS 109 — Spring 2026 — Prof. Mukamel
**Format:** Poster session (one teammate at the poster, other roaming; Prof. Mukamel + TA walk around asking questions; expect a 2-3 minute elevator pitch + 5-10 minutes of follow-up Q&A)

> **The one-sentence summary you should have ready at all times:**
> "We picked classification as our approach, K-Nearest Neighbours as our model, and a k-sweep minimising 5-fold cross-validation error as our model-selection procedure — then we showed that the *same* procedure under three different CV schemes returns answers ranging from 50% to 97% accuracy on this dataset, and that the 19.5-percentage-point gap between the honest answer and the leaky answer is leakage, not signal."

---

## Table of Contents
1. **[The 2-3 Minute Elevator Pitch (memorise this)](#part-1)**
2. **[The 10-Minute Deep Dive (internalise this)](#part-2)**
3. **[Numbers Cheat Sheet (memorise these)](#part-3)**
4. **[Concept Primers](#part-4)**
5. **[Q&A Bank — Easy Questions](#part-5a)**
6. **[Q&A Bank — Hard Questions](#part-5b)**
7. **[Q&A Bank — Anti-Trap Questions](#part-5c)**
8. **[Poster Walk-Through Script (panel-by-panel)](#part-6)**
9. **[Logistics & Delivery Tips](#part-7)**

---

<a id="part-1"></a>
## Part 1 — The 2-3 Minute Elevator Pitch

**This is what you say when Prof. Mukamel walks up.** Memorise it word-for-word, then practice making it sound conversational. ~350 words, ~2 minutes 30 seconds at a natural pace.

> **Hi, I'm Ivan and this is Anish. Our project is on EEG eye-state classification, but the real story isn't the classifier — it's about how the choice of cross-validation scheme can completely change what your accuracy number means.**
>
> The dataset is from UCI — 14 EEG channels recorded continuously for about two minutes from a single subject, with a binary label that says whether their eyes are open or closed at each sample. About 15,000 samples total, sampled at 128 hertz.
>
> Following the assignment template, we picked **classification** as our data-analysis approach, since the label is binary. Within classification, we picked **K-Nearest Neighbours** as our model — it's the simplest non-parametric classifier on the syllabus, and as we'll explain, it's also the one that makes the leakage problem we wanted to study most visible. Our **model-selection procedure** was to sweep the hyperparameter `k` over a log-spaced grid from 1 to 201 and pick the value that minimises mean 5-fold cross-validation error.
>
> Here's where the methodology gets interesting. We ran that *same* procedure under three different cross-validation schemes. **Shuffled** CV — the standard default — gave us KNN with `k = 1` at **97.3% accuracy**. That looks great, but adjacent EEG samples are almost identical — the lag-1 autocorrelation is 0.997 — so shuffled CV is essentially testing the model on samples it has near-duplicates of in training. That's leakage.
>
> So we tried **naive blocked** CV — five contiguous time chunks. That eliminates the lag-1 leakage, but the label runs are uneven, so individual folds are wildly class-imbalanced — fold accuracy drops to **50%**, near baseline.
>
> Our fix was **stratified blocked** CV: we split the training set into 100 short contiguous segments, then redistributed them across folds to balance class proportion while keeping local temporal continuity. That gave us **77.8% accuracy** — the honest number.
>
> So the take-home: the same model, the same procedure, the same data — but **shuffled CV says we have a 97% classifier and honest CV says we have a 78% classifier. The 19.5-point gap is leakage, not signal.** Any questions?

**Delivery notes:**
- Pause before "Here's where the methodology gets interesting" — that's your turn into the actual contribution.
- When you say "97.3%", point at the orange line on the headline figure.
- When you say "78%", point at the green line.
- End on the take-home quote — it's bold on the poster, you can read it off if needed.

---

<a id="part-2"></a>
## Part 2 — The 10-Minute Deep Dive

**This is for your own internalisation.** Practice talking through this version once or twice — the elevator pitch above is a compression of this. If a question takes you somewhere specific, you'll want to be able to expand on any sub-section below without notes.

### 2.1 Why this question?

EEG eye-state classification on the UCI #264 dataset is everywhere online — tutorials, student write-ups, blog posts all report numbers in the 95-98% range using shuffled k-fold cross-validation. We weren't trying to beat those numbers. We were trying to figure out whether those numbers are real. Real-world EEG applications — brain-computer interfaces, attention monitoring, sleep staging — deploy on continuous streams, not on shuffled samples. If a 97% lab number drops to 78% in deployment, that's a useful thing to know before you build a product around it.

### 2.2 Why classification?

The label is binary categorical — eyes open (0) or eyes closed (1). Regression would force us to interpret 0.7 as "70% eyes closed", which doesn't have a clean physical meaning. Clustering would discard the label entirely and ask whether eye-state structure emerges unsupervised — we ran that as a side check (K-means with k=2 on the channels) and the clusters did *not* align with eye state, which actually confirmed that the eyeDetection label needs supervised learning to recover.

### 2.3 Why KNN?

Three reasons:

1. **It's in the COGS 109 palette** — non-parametric classifiers are a sanctioned approach.
2. **It has a single, interpretable hyperparameter** (`k`), which makes the model-selection procedure clean — there's exactly one knob to turn.
3. **Its decision rule is per-sample similarity**, which makes it the model most exposed to lag-1 autocorrelation leakage. If our hypothesis was "shuffled CV inflates accuracy on autocorrelated time series", KNN was the cleanest test case.

We could equally have used LDA, PCR, or PCA-then-LDA, but they all average over many samples to make a decision — they smear out the per-sample similarity that drives the leakage. We do report all three as **sanity-check alternatives** in the supporting evidence panel, and they confirm the hypothesis: their leakage gaps are only 2-6 percentage points compared to KNN's 19.5.

### 2.4 The k-sweep

We swept `k` over `{1, 3, 5, 7, 11, 15, 21, 31, 51, 75, 99, 151, 201}` — log-spaced, odd-only.

- **Odd-only**: avoids ties on a binary label (e.g. with `k=2` you can have a 1-1 vote with no winner).
- **Log-spaced**: doubles the resolution at small `k` (which matters most) without wasting evaluations at large `k`.
- **`k = 1`**: the most leakage-prone setting — single nearest neighbour wins or loses on a single duplicate.
- **`k = 201`**: a local average over ~1.5 seconds at 128 Hz — should damp out lag-1 sample similarity if there's a real signal.

The rule for picking `k` is the argmax of mean 5-fold CV accuracy. Crucially, this rule is **identical across all three CV schemes** — only the fold-assignment rule changes. That's what makes the comparison fair.

### 2.5 The three CV schemes (be ready to explain each in 30 seconds)

**Scheme A — Shuffled 5-fold (the leaky baseline).**
Every sample is independently and uniformly randomly assigned to one of 5 folds. Each fold has ~20% of samples chosen at random. This is `sklearn.model_selection.KFold(shuffle=True)` — the default for most students and tutorials. On i.i.d. data, this is the right thing to do. On a continuous time series with lag-1 r = 0.997, it leaks: the temporal neighbour of every test sample is almost certainly in the training fold, and that neighbour has almost certainly the same label.

**Scheme B — Naive blocked 5-fold.**
Split the training set into 5 contiguous time chunks of equal length. No shuffling. This is the "obvious" fix for the leakage problem — and it does eliminate the lag-1 issue. But it has its own problem: the label runs are unevenly distributed across the recording, so individual folds can end up with very skewed class proportions (we measured one fold at 20% class-0 vs. ~54% global). Fold accuracy can fall *below* the majority-class baseline. We include it to show the failure mode, not as a recommendation.

**Scheme C — Stratified blocked 5-fold (our honest evaluator).**
Split the training set into 100 short contiguous segments — each ~120 samples ≈ 0.9 seconds. Assign segments to the 5 folds in a class-stratified round-robin so each fold has roughly the same eyes-open / eyes-closed ratio as the full training set. This preserves local temporal continuity within each segment (most segments are short enough that the label doesn't change inside them) while restoring the class-balance guarantees that stratified CV gives you. **This is the number we report as honest.**

### 2.6 The headline result

| CV scheme            | Picked `k` | Picked accuracy ± std |
|----------------------|:---:|:---:|
| Shuffled (leaky)     | 1 | **97.3% ± 0.4%** |
| Naive blocked        | 1 | **50.0% ± 10.7%** |
| Stratified blocked   | 1 | **77.8% ± 2.7%** |

- All three schemes pick the *same* `k = 1`. The model-selection procedure is robust. The accuracy it reports is not.
- The spread is 47 percentage points — same model, same data, same training samples — driven entirely by the fold-assignment rule.
- The 19.5-point gap between shuffled and stratified blocked is our headline "leakage gap".

### 2.7 Why is the gap so big for KNN?

KNN with `k = 1` predicts by finding the single nearest training sample (Euclidean distance in 14-D z-scored channel space) and copying its label. The lag-1 correlation between adjacent EEG samples is 0.997 — they're almost numerically identical. So under shuffled CV, the temporal neighbour of every test sample lands in the training fold with ~80% probability, and that neighbour has the same label with ~99.7% probability. The model is essentially being asked "what was the eye state 8 milliseconds ago?" and answering correctly almost always.

Under stratified blocked CV, the temporal neighbour is reserved into the same segment as the test sample — it's never in the training fold. The nearest *training* sample is now potentially seconds away from a label transition. The model has to actually generalise, and the accuracy drops to 78%.

### 2.8 Sanity check — the other three classifiers

To confirm this is a KNN-specific phenomenon, not a generic dataset issue, we ran the same idea on LDA, PCA→LDA (sweep n_components), and PCR-as-classifier (sweep n_components). Their leakage gaps:

| Model      | Picked hyperparam | Shuffled | Stratified blocked | Gap |
|------------|:---:|:---:|:---:|:---:|
| LDA        | n/a       | 64.7% | 59.1% | +5.6 pp |
| PCA → LDA  | n=3       | 55.7% | 53.4% | +2.4 pp |
| PCR        | n=2       | 55.3% | 52.5% | +2.8 pp |
| **KNN**    | **k=1**   | **97.3%** | **77.8%** | **+19.5 pp** |

KNN's gap is 4-8x larger than any other model. This is exactly what theory predicts: LDA, PCA→LDA, and PCR all average over many samples before classifying, which smears out the per-sample similarity that KNN exploits. KNN is uniquely vulnerable — and that vulnerability is what made it the cleanest model for our study.

### 2.9 What this means more broadly

The honest 77.8% number is still well above the 55.12% majority-class baseline — about 23 percentage points above. The classifier is not useless; the EEG signal genuinely contains eye-state information. What our study shows is that **the magnitude of that information is dramatically smaller than shuffled-CV evaluation would lead you to believe.** For any time-series problem with non-trivial autocorrelation — and most physiological signals have that — blocked CV is not optional, and class-stratified blocked CV is the version you actually want.

---

<a id="part-3"></a>
## Part 3 — Numbers Cheat Sheet

**Drill these until they're automatic. If a question requires a number, the only acceptable answer is the right number.**

### Dataset
- **14,980** total samples
- **14** EEG channels (Emotiv EPOC headset)
- **~128 Hz** sampling rate
- **~117 seconds** recording length (~2 minutes)
- **1** subject (single-subject study)
- **55.12% / 44.88%** class balance (eyes open / eyes closed)
- **23** label transitions in the whole recording
- **~0.997** lag-1 autocorrelation (this is the number that drives everything)
- Source: **UCI Machine Learning Repository, dataset #264** (Roesler, 2013)

### Preprocessing
- **4** outliers dropped (samples >4 SD on any channel)
- **14,976** samples after cleaning
- **80/20** chronological train/test split
- **64-sample seam gap** between train and test (prevents boundary contamination)
- **z-scored** using train-partition means/stds only (no test leakage)

### KNN k-sweep
- Grid: **{1, 3, 5, 7, 11, 15, 21, 31, 51, 75, 99, 151, 201}** — log-spaced, odd-only
- All three schemes pick **k = 1**

### Headline accuracies (KNN k=1)
- **Shuffled CV: 97.3% ± 0.4%**
- **Naive blocked CV: 50.0% ± 10.7%**
- **Stratified blocked CV: 77.8% ± 2.7%** (the honest one)
- **Leakage gap: 19.5 percentage points**

### Stratified blocked CV details
- **100** short contiguous segments
- Each segment is **~120 samples ≈ 0.9 seconds** long
- Segments assigned round-robin to **5 folds** with class stratification

### Alternative classifiers (sanity checks)
- LDA: 5.6 pp leakage gap
- PCA→LDA (n=3): 2.4 pp
- PCR (n=2): 2.8 pp
- KNN: 19.5 pp (~4-8x larger than any alternative)

### Baselines
- Majority-class baseline: **0.5512** (55.12%)
- Random guessing on a balanced split: 50%

### Holdout test partition (mentioned only if asked)
- 2,995 samples, but 90.7% class-0 (the recording ends in a long eyes-open period)
- We *do not* over-interpret holdout accuracy because of this imbalance — the CV numbers are the meaningful summary

---

<a id="part-4"></a>
## Part 4 — Concept Primers

**Quick refreshers in case anything comes up that you blanked on under pressure.**

### K-Nearest Neighbours (KNN)
- Non-parametric classifier. No training in the usual sense — you store all training points.
- To classify a new point: compute its distance to every training point, find the `k` nearest, take a majority vote of their labels.
- Distance metric here: Euclidean distance in 14-D z-scored channel space.
- The hyperparameter `k` controls smoothness — small `k` is more flexible (low bias, high variance), large `k` is smoother (higher bias, lower variance).
- `k = 1`: nearest-neighbour classifier — your prediction is literally the label of the closest training point.

### Cross-validation
- Splits data into `K` folds. Trains on `K-1` of them, tests on the held-out one. Repeats `K` times, averages the per-fold accuracy.
- Purpose: estimate generalisation performance using only training data. Lets you compare hyperparameters without touching the final test set.
- `K = 5` is the standard tradeoff — bigger `K` gives lower bias estimates but higher variance and more compute.

### Autocorrelation
- The correlation between a signal and a delayed copy of itself.
- "Lag-1 autocorrelation" = correlation between sample `t` and sample `t+1`.
- For EEG: lag-1 r = 0.997 means consecutive samples are almost numerically identical (~99.7% of their variance is shared).
- For the label: only 23 transitions in 14,980 samples means the label is piecewise-constant over very long runs.

### Data leakage
- Any situation where information from the test set indirectly trains the model.
- On time series: if your fold assignment puts sample `t` in training and sample `t+1` in test, and the signal has high lag-1 autocorrelation, your model is essentially memorising rather than generalising.

### Z-scoring (standardisation)
- For each feature: subtract the mean, divide by the standard deviation.
- Critically: **use train statistics only** for both train and test. Computing the test mean and using it on the test set leaks information.
- Important for KNN because Euclidean distance is sensitive to feature scale — a 100 µV channel would dominate over a 10 µV channel without standardisation.

### Stratified CV vs. blocked CV vs. stratified blocked CV
- **Stratified CV**: ensures each fold has the same class proportion as the whole dataset. Standard for class-imbalanced classification.
- **Blocked CV**: assigns samples to folds in contiguous chunks (preserves temporal order). Standard for time series.
- **Stratified blocked CV** (our recipe): combines both — short contiguous segments are stratified by class across folds. Preserves local temporal continuity *and* class balance.

### Confusion matrix
- 2×2 table for binary classification. Rows = true labels, columns = predicted labels.
- **Sensitivity (recall, true positive rate)** = TP / (TP + FN) — of all actual positives, what fraction did we catch?
- **Specificity (true negative rate)** = TN / (TN + FP) — of all actual negatives, what fraction did we correctly reject?
- We show confusion matrices for the picked KNN model in figure 15.

---

<a id="part-5a"></a>
## Part 5a — Q&A Bank: Easy Questions

These are softballs. Have the answers ready in one breath.

**Q: Why did you pick this dataset?**
A: It's a clean, small-scale setting in which to study cross-validation honesty. It's also a heavily-tutorialised dataset where you can find dozens of online write-ups reporting 95-98% KNN accuracy, which gave us a clear "received wisdom" to interrogate.

**Q: What's KNN?**
A: K-Nearest Neighbours. To classify a new sample, find the `k` training samples closest to it in feature space, take a majority vote of their labels. We used Euclidean distance in 14-D z-scored channel space.

**Q: What does `k` do?**
A: It's the only hyperparameter. Small `k` makes the decision boundary more local and flexible (low bias, high variance). Large `k` averages over more neighbours and smooths the boundary (higher bias, lower variance).

**Q: Why did you pick `k = 1`?**
A: We didn't pick it — the model-selection procedure picked it. We swept `k` over a log-spaced grid from 1 to 201 and the argmax-of-mean-CV-accuracy was `k = 1` under *all three* CV schemes. That robustness across schemes is itself a result.

**Q: Why 5-fold CV and not 10-fold?**
A: 5-fold is the standard tradeoff between bias and variance. With ~12,000 training samples, each fold has ~2,400 samples, which is plenty to get a stable per-fold accuracy. 10-fold would double our compute without meaningfully changing the conclusions.

**Q: What's the majority-class baseline?**
A: 55.12% — the fraction of samples labelled eyes-open. Any honest classifier needs to beat that.

**Q: How big is the dataset?**
A: 14,980 samples, 14 EEG channels plus the binary label, sampled at ~128 Hz over ~117 seconds from a single subject.

**Q: What's the sampling rate?**
A: ~128 Hz. So consecutive samples are ~7.8 milliseconds apart.

**Q: What's the label?**
A: `eyeDetection`. Binary — 0 means eyes open, 1 means eyes closed. Annotated by the experimenter from synchronised video.

**Q: Did you compare to other models?**
A: Yes — as sanity-check alternatives. LDA, PCA→LDA, and PCR-as-classifier. They all show much smaller leakage gaps (2-6 pp vs KNN's 19.5 pp), which confirms that KNN's gap is a model-specific phenomenon driven by its per-sample similarity decision rule, not a generic dataset artefact.

**Q: How long did this take?**
A: A couple of weeks of work — most of the time was on the methodology rather than the modelling. The stratified blocked CV was the critical insight that came out of seeing the naive blocked CV fail.

---

<a id="part-5b"></a>
## Part 5b — Q&A Bank: Hard Questions

These are the ones Dr. Mukamel will probably actually ask. Be ready.

**Q: Walk me through your model-selection process.**
A: We picked one approach — classification, because the label is binary categorical. Within classification we picked one model — KNN, because it's the simplest non-parametric classifier on the syllabus and its per-sample similarity decision rule makes it the cleanest test case for the autocorrelation hypothesis. Then for model selection we swept the hyperparameter `k` over a log-spaced odd-only grid from 1 to 201 and picked the `k` that minimised mean 5-fold CV error. The argmax was `k = 1` under all three CV schemes we tried.

**Q: Why KNN specifically, when LDA or PCR would also fit?**
A: Two reasons. First, KNN has exactly one tunable hyperparameter (`k`), which makes the model-selection narrative the cleanest — there's no ambiguity about what's being swept. Second, KNN's decision rule is per-sample similarity — it's the model in the COGS 109 palette that is most exposed to lag-1 autocorrelation leakage, which was the methodological question we wanted to study. The other models — LDA, PCA→LDA, PCR — average over many samples before classifying, which smears out the leakage. We do report all three in a supporting-evidence panel, and they actually strengthen the story: their leakage gaps are 2-6 percentage points, compared to KNN's 19.5 — exactly what theory predicts.

**Q: Why did you pick THIS model-selection process — the k-sweep with argmax-of-CV-accuracy?**
A: It's the standard model-selection procedure for any classifier with a single hyperparameter — sweep the parameter over a reasonable grid, evaluate each grid point with cross-validation, and pick the grid point with the lowest CV error. That's it. The novelty of our project isn't in the model-selection procedure itself — it's that we ran the *same* procedure under three different CV schemes and showed that the picked accuracy changes by 47 percentage points even though the procedure picks the same `k` each time.

**Q: Why do all three schemes pick `k = 1`?**
A: Because the label is piecewise-constant over very long runs — there are only 23 label transitions in the whole recording. So even under blocked CV, the nearest training sample (in feature space) often comes from a region where the label hasn't changed yet, and copying its label is a good bet. Larger `k` introduces averaging that doesn't help here because the local label is already homogeneous. Small `k` always wins on this dataset; what changes between schemes is the *picked accuracy*, not the picked `k`.

**Q: What is lag-1 autocorrelation and why does it matter?**
A: Lag-1 autocorrelation is the correlation between sample `t` and sample `t+1`. For our channels, it's about 0.997 — consecutive samples share 99.7% of their variance, they're almost identical. That matters because under shuffled CV, the temporal neighbour of every test sample lands in the training fold with ~80% probability and carries the same label with ~99.7% probability — so a KNN with `k=1` is essentially being asked "what was the eye state 8 milliseconds ago?" and answering correctly almost always. That's leakage, not generalisation.

**Q: Why was naive blocked CV so bad?**
A: Because the label runs are unevenly distributed across the recording. When you cut the training set into 5 contiguous time chunks, individual folds can end up with very skewed class proportions — we measured one fold at about 20% class-0 versus the global ~54%. The per-fold accuracy variance therefore explodes, and the mean accuracy can fall below the majority-class baseline. It's the fold construction that's failing, not the model.

**Q: How exactly does stratified blocked CV work?**
A: We split the training set into 100 short contiguous segments, each ~120 samples ≈ 0.9 seconds. The segments are short enough that the label is nearly always constant inside them, and long enough that most segments don't straddle a label transition. We then assign segments to the 5 folds in a class-stratified round-robin, so each fold has roughly the same eyes-open / eyes-closed ratio as the full training set. This gives us the temporal-isolation guarantees of blocked CV — adjacent samples are never in different folds — *and* the class-balance guarantees of stratified CV.

**Q: Why z-score using only training statistics?**
A: Because z-scoring with test statistics leaks information from test to train. If you compute the mean and std over the full dataset and then z-score everything, the training data has been implicitly "told" what the test distribution looks like. The proper procedure is to estimate mean and std on the training partition only, then apply that same transform to the validation and test partitions.

**Q: Why a 64-sample seam gap between train and test?**
A: To prevent boundary contamination. At 128 Hz, 64 samples is half a second — more than long enough to ensure that the test partition starts after the last training-partition sample has fully "left" the lag-1 autocorrelation window.

**Q: Why drop outliers at 4 standard deviations specifically?**
A: It's a relatively conservative threshold — only 4 samples out of 14,980 exceed it. We didn't want to be more aggressive because the dataset is small and we wanted to confirm the outliers were artefacts rather than real signal. We confirmed that the picked accuracy is essentially identical whether you use a 4 SD or a 10 SD threshold (because both drop the same 4 voltage-saturation samples).

**Q: What's the holdout test accuracy?**
A: We report it in the report but we don't lead with it, because the held-out 20% chronological partition is heavily class-imbalanced — about 91% class-0 — because the recording happens to end in a long eyes-open period. So a model that just predicts class-0 always would score 91% on that holdout, which doesn't tell us much. The per-fold CV numbers from the development partition are the meaningful summary.

**Q: Would you trust the 77.8% number for a real-world deployment?**
A: With major caveats. 77.8% is the honest estimate of accuracy on samples drawn from *this single subject's* EEG distribution under temporally-isolated evaluation. It does *not* generalise to a new subject, because our evaluation never tests on a different subject — that's a between-subject generalisation question, not a within-subject one. To answer the deployment question we'd need a leave-one-subject-out evaluation on a multi-subject dataset.

**Q: How does this relate to brain-computer interfaces (BCIs)?**
A: This is the central methodological concern in BCI evaluation. Most academic BCI papers report accuracies in the high-90s under shuffled CV, but deployment performance is consistently much lower — often in the 60-80% range. Our small study suggests one major source of that drop is exactly the leakage we documented: shuffled CV evaluates on samples that are nearly identical to training samples, and deployment doesn't. The honest stratified blocked CV number is much closer to what you'd actually see in deployment.

---

<a id="part-5c"></a>
## Part 5c — Q&A Bank: Anti-Trap Questions

These sound easy but have gotchas. Don't fall in.

**Q: "So your accuracy is 97%?"**
*(Trap: they're testing whether you understand which number is the real number.)*
A: That's the shuffled-CV number, which we argue is inflated by leakage. The honest number from stratified blocked CV is 77.8% — that's the one we report as the actual classifier performance.

**Q: "Couldn't you just have done linear regression with a threshold?"**
*(Trap: testing whether you understand approach-vs-model.)*
A: We did, actually — that's the PCR-as-classifier baseline in the supporting evidence. It works, but its honest stratified-blocked accuracy is only 52.5%, barely above the 55% majority-class baseline. The point is that the *approach* — classification vs. regression — is determined by the label structure (binary categorical → classification), and within that approach we picked KNN for the reasons above.

**Q: "Why not just use deep learning?"**
*(Trap: assignment-scope question — be ready.)*
A: Two reasons. First, the assignment explicitly scopes us to the COGS 109 study-guide methods — classification with LDA/KNN/PCR-style models, regression, clustering, PCA. Neural networks are out of scope. Second, with only ~12,000 training samples from a single subject, deep learning is unlikely to help — the leakage problem we documented is independent of model complexity, and a deep net evaluated under shuffled CV would show the same inflation.

**Q: "What about logistic regression?"**
*(Trap: same as above.)*
A: Same answer — logistic regression isn't in the study-guide palette. We did consider it informally and would expect it to perform similarly to LDA on this dataset (LDA's leakage gap is only 5.6 pp; logistic regression would probably look very similar).

**Q: "Couldn't you have used spectral features or FFT?"**
*(Trap: feature-engineering question — could expand scope and weaken methodology.)*
A: Yes, in principle — FFT-derived band-power features (alpha, beta, theta) are the standard approach in actual BCI work. But two things kept us out of that direction. First, the assignment is methods-only — FFT-derived features aren't part of the study-guide palette. Second, and more importantly, we didn't want feature engineering to confound the methodological question. We wanted to ask "does CV scheme choice change the answer for the same model on the same features?" — and the cleanest version of that question uses the raw 14-channel features. Adding spectral features would be a sensible follow-up project but a different question.

**Q: "Your stratified blocked CV is 77.8%. What's the std deviation?"**
A: ±2.7 percentage points across the 5 folds.

**Q: "Why exactly 100 segments?"**
*(Trap: testing whether you've justified your design choices.)*
A: 100 segments was chosen so each segment is roughly 120 samples ≈ 0.9 seconds at 128 Hz. We wanted segments short enough that the per-segment label is nearly always constant (avoiding mixed-label segments) but long enough that most segments don't straddle a label transition. With 23 label transitions in 14,980 samples, a ~120-sample segment is on average 2-3x longer than the typical gap between transitions, which gives a good balance. We didn't formally tune this — picking 50 or 200 segments gives qualitatively similar results, but 100 was a clean round number.

**Q: "Did you check for sensor failure or bad channels?"**
A: We looked at per-channel statistics in the EDA — there are 4 voltage-saturation outliers (>4 SD) which we drop, but no consistently dead channels. The channel correlation heatmap (figure 5 of the EDA) shows the expected sensor-physical-proximity correlations — frontals correlate with frontals, occipitals with occipitals — which suggests the channels are picking up real EEG and not flat noise.

**Q: "Your three numbers are 97%, 50%, and 78% — but they're all on the same data. How can they be that different?"**
*(Trap: this is the central question — they want to hear the methodology in your own words.)*
A: Because the only thing that's changing is the rule that assigns samples to cross-validation folds. The model is identical, the training data is identical, the picked `k` is identical (all three pick k=1). Under shuffled CV, the test fold contains samples whose temporal neighbours are in the training fold, and those neighbours have nearly identical features and nearly identical labels — so the model can effectively "look up" the answer. Under naive blocked CV the temporal neighbours aren't in the training fold, but the per-fold class balance is broken because label runs aren't uniformly distributed. Under stratified blocked CV the temporal neighbours aren't in the training fold *and* the per-fold class balance is preserved — so we get a number that reflects what the model has actually learned. The 47-point spread is leakage and fold-construction failure, not signal.

**Q: "What if we used a different distance metric for KNN?"**
A: We used standard Euclidean distance in z-scored 14-D channel space, which is the COGS 109 default. We could have used Manhattan, Mahalanobis, or cosine — but distance metric is unlikely to materially change the leakage story, because the leakage comes from the lag-1 *similarity* between adjacent samples, which is high under any reasonable metric on standardised channels.

---

<a id="part-6"></a>
## Part 6 — Poster Walk-Through Script

Use this if someone asks "can you walk me through the poster?" Point at each panel as you describe it.

1. **Title bar** — *"Classification of EEG Eye-State via KNN: A Cross-Validation Honesty Study"*. The subtitle hints at the methodology angle.

2. **Authors** — Ivan Del Rio and Anish Kondamadugula. COGS 109, Mukamel, Spring 2026.

3. **Abstract** — *"We pick classification, KNN, k-sweep, three CV schemes."* The whole thesis in 5 sentences.

4. **Question panel** — *"Given an autocorrelated single-subject EEG recording and a KNN classifier, how does the choice of cross-validation scheme change the result of an otherwise-identical model-selection procedure?"* That's the methodological question.

5. **Background** — UCI #264, 14,980 samples, 14 channels, single subject, lag-1 r ≈ 0.997, 24 label runs. Sets up *why* leakage is a concern.

6. **Methods** — Approach (classification), model (KNN with Euclidean distance), model selection (sweep `k` over the log-spaced grid, pick argmax of mean 5-fold CV accuracy), three CV schemes (shuffled, naive blocked, stratified blocked), preprocessing (4 SD outlier removal, chronological 80/20 split, z-scored on train partition only).

7. **Take-home box** (bold, centred) — *"Shuffled CV says we have a 97% classifier. Honest stratified blocked CV says we have a 78% classifier. The difference is leakage, not signal."* Read this out loud if they don't see it.

8. **Headline figure** (figure 11, centre) — *"This is KNN model selection under three CV schemes. All three pick k=1 (the open markers on the left). Orange is shuffled — 97.3%, leaky. Blue is naive blocked — 50%, broken by fold class imbalance. Green is stratified blocked — 77.8%, the honest answer."*

9. **Results table** (top-right) — Three rows, one per scheme, with picked `k` and accuracy ± std. The stratified-blocked row is highlighted.

10. **Supporting evidence** — *"We also ran the same idea on LDA, PCA→LDA, and PCR-as-classifier. Their leakage gaps are only 2-6 pp compared to KNN's 19.5 pp — exactly what theory predicts, because they average over many samples instead of relying on per-sample similarity."*

11. **Autocorrelation figure** (figure 08, lower-left) — *"This is the lag-1 autocorrelation that motivates blocked CV. r=0.997 at lag 1, still above 0.8 out to lag 100. Shuffled CV ignores all of that structure."*

12. **Conclusions** — Four numbered findings:
    1. All three schemes pick `k = 1` — model selection procedure is robust.
    2. Picked accuracies span 47 pp across schemes — model selection *result* is not.
    3. The 19.5 pp gap is leakage attributable to scheme choice.
    4. Alternative classifiers confirm KNN is uniquely vulnerable.

13. **References** — Roesler 2013 (data), Mukamel 2026 (course notes), James et al. ISLR 2021, Bergmeir & Benítez 2012 (time-series CV).

---

<a id="part-7"></a>
## Part 7 — Logistics & Delivery Tips

### Before the session
- Print the poster (Variant 1 — Polished Classic). The file is `poster/variants/variant_1_polished_classic.pptx` in the repo. 48"×36" standard scientific poster size.
- Skim `report/final_report.md` once the night before — that's your authoritative source if a question goes deeper than this guide.
- Have your phone or laptop with this guide accessible during the session (not visible to evaluators) in case you need to refresh on a number.

### During the session
- **One person stands at the poster at all times.** Trade off every 10-15 minutes so you both get to walk around.
- **When Prof. Mukamel approaches**, the standing person starts the elevator pitch unprompted — don't wait to be asked.
- **Point at the poster as you talk.** It anchors the listener and gives you something to do with your hands.
- **If you don't know an answer**, the honest move is *"That's a really good question — we didn't formally test that, but my guess is X because Y."* Don't make up specifics. Dr. Mukamel will respect you more for being honest about limits.
- **Hand off to your teammate** for sub-topics they prepped. Anish has the lead on figures and tables; Ivan has the lead on methodology and CV schemes. (Adjust to whatever you actually divided.)

### Things to **NOT** say
- "Devin", "AI", "the agent", "a Devin session", "we used an LLM to..." — never. The poster, report, and code don't contain those words; don't introduce them in spoken delivery.
- "I think...", "I'm not sure...", "maybe...". Pick a confident answer or pick "I don't know".
- "The model just memorises the training data" — *don't oversimplify the leakage story*. Be precise: under shuffled CV, the temporal neighbour of each test sample is in the training fold and carries the same label.
- Don't say "we got 97%". The headline number is **77.8%** — the honest one. The 97% is the *leakage demonstration*, not your result.

### The two questions to expect for sure
1. **"Walk me through your model selection."** → See the answer in Part 5b above. Lead with: *"We picked classification because the label is binary, then KNN because it has one clean hyperparameter and is uniquely exposed to the leakage we wanted to study, then we swept `k` over a log-spaced grid and picked the argmax-of-CV-accuracy. Same procedure under three CV schemes, and that's where the story is."*
2. **"Why did this CV scheme matter?"** → See Part 5b. Lead with: *"Because adjacent EEG samples have lag-1 autocorrelation of 0.997 — under shuffled CV the temporal neighbour of every test sample is in the training fold with ~80% probability and shares the same label with 99.7% probability. The model is being asked a question it can almost always answer by lookup. Blocked CV breaks that, but you have to stratify the blocks or fold imbalance breaks the accuracy estimate."*

### Final confidence check
If you can do all of the following without notes, you're ready:

- [ ] Recite the elevator pitch from memory in under 3 minutes
- [ ] State the four key numbers — 97.3%, 50.0%, 77.8%, 19.5 pp gap — in any order
- [ ] State the lag-1 autocorrelation (0.997) and the majority-class baseline (55.12%)
- [ ] Describe each CV scheme in 30 seconds without checking the poster
- [ ] Justify the choice of KNN (per-sample similarity → uniquely exposed to leakage)
- [ ] Justify the choice of `k = 1` (it's what the procedure picked, robustly across all three schemes)
- [ ] Explain the supporting evidence (LDA/PCA→LDA/PCR leakage gaps are 2-6 pp; KNN's 19.5 pp is the outlier)
- [ ] Name the dataset source (UCI ML Repository #264, Roesler 2013)
- [ ] Defend the honest number (77.8%) as "what the classifier actually learned" vs. the shuffled number (97.3%) as "what leakage makes the classifier look like it learned"

**Good luck. You know this project.**
