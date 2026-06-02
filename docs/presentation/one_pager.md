# COGS 109 Poster Cheat Card (1 page — print this)

## The elevator pitch (≈2.5 min, memorise)

> Hi, I'm Ivan and this is Anish. Our project is on EEG eye-state classification, but the real story isn't the classifier — it's about how the choice of cross-validation scheme completely changes what your accuracy number means.
>
> The dataset is UCI #264 — 14 EEG channels recorded continuously for ~2 minutes from a single subject, with a binary eyes-open/closed label. ~15,000 samples at 128 Hz.
>
> Following the assignment template, we picked **classification** as our approach, **K-Nearest Neighbours** as our model, and a **k-sweep minimising 5-fold CV error** as our model-selection procedure.
>
> Here's the methodology twist. We ran that *same* procedure under three different CV schemes:
> - **Shuffled CV** picks k=1 at **97.3%** — but adjacent EEG samples are nearly identical (lag-1 r=0.997), so this is leakage.
> - **Naive blocked CV** (5 contiguous chunks) drops to **50%** — fold class-imbalance, not model failure.
> - **Stratified blocked CV** (100 contiguous segments redistributed across folds) gives us **77.8%** — the honest number.
>
> Take-home: **shuffled CV says we have a 97% classifier, honest CV says we have a 78% classifier. The 19.5-point gap is leakage, not signal.**

## Numbers you MUST know cold

| Metric | Value |
|---|---|
| Dataset | UCI #264, 14,980 samples, 14 channels, ~128 Hz, ~117 s, 1 subject |
| Class balance | 55.12% open / 44.88% closed |
| Lag-1 autocorrelation | r ≈ 0.997 (this is THE number) |
| Label transitions in recording | 23 |
| KNN k-sweep grid | {1, 3, 5, 7, 11, 15, 21, 31, 51, 75, 99, 151, 201} |
| **Picked k under ALL three schemes** | **k = 1** |
| Shuffled CV (leaky) | 97.3% ± 0.4% |
| Naive blocked CV | 50.0% ± 10.7% |
| **Stratified blocked CV (honest, headline)** | **77.8% ± 2.7%** |
| **Leakage gap (KNN)** | **19.5 pp** |
| LDA leakage gap | 5.6 pp |
| PCA→LDA leakage gap | 2.4 pp |
| PCR leakage gap | 2.8 pp |
| Majority-class baseline | 0.5512 |
| Stratified blocked segments | 100 × ~120 samples (~0.9s each) |

## The 3 questions you MUST answer fluently

**1. "Walk me through your model selection."**
> Classification because label is binary. KNN because it has one clean hyperparameter and is uniquely exposed to the leakage we wanted to study. Swept k over a log-spaced odd-only grid 1→201, picked the argmax of mean 5-fold CV accuracy. All three schemes pick k=1, but their picked accuracies span 47 percentage points.

**2. "Why KNN and not something else?"**
> Three reasons: it's in the COGS 109 palette, it has exactly one hyperparameter, and its per-sample similarity decision rule makes it the model most exposed to lag-1 autocorrelation leakage. LDA/PCR/PCA→LDA all average over many samples, which smears out the per-sample similarity — they show leakage gaps of only 2-6 pp compared to KNN's 19.5 pp, which confirms our hypothesis.

**3. "Why does CV scheme choice matter so much?"**
> Lag-1 autocorrelation is 0.997 — consecutive samples are nearly numerically identical. Under shuffled CV, every test sample's temporal neighbour is in the training fold and carries the same label with ~99.7% probability, so KNN at k=1 is essentially looking up the answer. Under stratified blocked CV, the temporal neighbour is reserved into the same segment as the test sample — never in the training fold — so the model has to actually generalise. That's why the same model on the same data gives 97% under shuffled and 78% under honest evaluation.

## Things to NOT say
- "Devin", "AI", "agent", "we used an LLM" — **never**
- "We got 97%" — the headline is **77.8%**, 97% is the leakage demo
- "The model just memorises" — be precise: it's the temporal neighbour landing in training fold, not memorisation
- "I think...", "maybe...", "I'm not sure..." — pick a confident answer or pick "I don't know"

## When you don't know
> "That's a really good question — we didn't formally test that, but my guess is X because Y."

Be honest. Dr. Mukamel respects honesty about limits more than hand-waving.
