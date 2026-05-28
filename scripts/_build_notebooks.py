#!/usr/bin/env python3
"""Helper that authors the canonical .ipynb files from inline source.

This is the single source of truth for notebook content. Re-running this
regenerates ``notebooks/00_fetch_data.ipynb`` and ``notebooks/01_eda.ipynb``
with empty outputs; subsequent ``jupyter nbconvert --execute --inplace``
calls populate the outputs for the committed artifacts.

Running this script is optional — the .ipynb files are committed and the
pipeline does not depend on it. It exists so that the notebook source is
diffable in code review without wading through JSON cell metadata.
"""

from __future__ import annotations

import os

import nbformat as nbf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NB_DIR = os.path.join(REPO_ROOT, "notebooks")
os.makedirs(NB_DIR, exist_ok=True)


def _nb(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    return nb


def _md(text: str):
    return nbf.v4.new_markdown_cell(text)


def _code(text: str):
    return nbf.v4.new_code_cell(text)


def build_fetch_notebook():
    cells = [
        _md(
            "# 00 — Fetch the EEG eye-state dataset\n"
            "\n"
            "This notebook downloads the UCI EEG Eye State dataset (repository\n"
            "id `264`), asserts the expected shape and label distribution, writes\n"
            "the canonical raw CSV to `data/raw/eeg_eye_state.csv`, and records a\n"
            "SHA-256 manifest. It is idempotent: if the CSV already exists locally\n"
            "the download step is skipped.\n"
            "\n"
            "Source: <https://archive.ics.uci.edu/dataset/264/eeg+eye+state>\n"
        ),
        _code(
            "import hashlib, json, os, sys\n"
            "from datetime import datetime, timezone\n"
            "\n"
            "REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..')) if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()\n"
            "if REPO_ROOT not in sys.path:\n"
            "    sys.path.insert(0, REPO_ROOT)\n"
            "\n"
            "RAW_DIR = os.path.join(REPO_ROOT, 'data', 'raw')\n"
            "RAW_CSV = os.path.join(RAW_DIR, 'eeg_eye_state.csv')\n"
            "MANIFEST = os.path.join(RAW_DIR, 'manifest.json')\n"
            "os.makedirs(RAW_DIR, exist_ok=True)\n"
            "RAW_CSV, MANIFEST"
        ),
        _md("## Load (or skip if already on disk)"),
        _code(
            "if os.path.exists(RAW_CSV):\n"
            "    import pandas as pd\n"
            "    df = pd.read_csv(RAW_CSV)\n"
            "    print(f'Loaded cached raw CSV: {RAW_CSV}  shape={df.shape}')\n"
            "else:\n"
            "    from ucimlrepo import fetch_ucirepo\n"
            "    import pandas as pd\n"
            "    repo = fetch_ucirepo(id=264)\n"
            "    df = pd.concat([repo.data.features, repo.data.targets], axis=1)\n"
            "    df.to_csv(RAW_CSV, index=False)\n"
            "    print(f'Downloaded UCI #264 and wrote {RAW_CSV}  shape={df.shape}')\n"
        ),
        _md("## Verify shape, columns, class counts, and NaN-freeness"),
        _code(
            "from src.data import CHANNELS, LABEL_COL\n"
            "\n"
            "assert df.shape == (14980, 15), f'unexpected shape {df.shape}'\n"
            "assert list(df.columns[:14]) == list(CHANNELS), 'channel order mismatch'\n"
            "assert LABEL_COL in df.columns, 'missing eyeDetection label column'\n"
            "assert df.isna().sum().sum() == 0, 'unexpected NaNs in raw data'\n"
            "class_counts = df[LABEL_COL].value_counts().to_dict()\n"
            "assert class_counts == {0: 8257, 1: 6723}, f'class balance mismatch: {class_counts}'\n"
            "class_counts"
        ),
        _md("## Write the SHA-256 manifest"),
        _code(
            "with open(RAW_CSV, 'rb') as f:\n"
            "    sha = hashlib.sha256(f.read()).hexdigest()\n"
            "\n"
            "manifest = {\n"
            "    'sha256': sha,\n"
            "    'downloaded_at': datetime.now(timezone.utc).isoformat(),\n"
            "    'source_url': 'https://archive.ics.uci.edu/static/public/264/eeg+eye+state.zip',\n"
            "    'n_rows': int(df.shape[0]),\n"
            "    'n_cols': int(df.shape[1]),\n"
            "    'channels': list(CHANNELS),\n"
            "    'label_column': LABEL_COL,\n"
            "    'class_counts': {str(k): int(v) for k, v in class_counts.items()},\n"
            "}\n"
            "with open(MANIFEST, 'w') as f:\n"
            "    json.dump(manifest, f, indent=2, sort_keys=True)\n"
            "manifest"
        ),
        _md(
            "Data loaded successfully. The 14,980 × 15 EEG eye-state dataset has\n"
            "been written to `data/raw/eeg_eye_state.csv` along with a SHA-256\n"
            "manifest. The label balance (0: 8257, 1: 6723) and channel order\n"
            "match the UCI Machine Learning Repository entry for dataset #264.\n"
            "Downstream code (the `scripts/preprocess.py` CLI and the\n"
            "`notebooks/01_eda.ipynb` analysis) reads from this canonical CSV.\n"
        ),
    ]
    nb = _nb(cells)
    nbf.write(nb, os.path.join(NB_DIR, "00_fetch_data.ipynb"))
    print("wrote notebooks/00_fetch_data.ipynb")


EDA_CODE_SETUP = r"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..')) if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data import CHANNELS, LABEL_COL, fetch_raw, clean, split_chronological, fit_scaler, apply_scaler
from src.cv import blocked_kfold_indices
from src.plotting import apply_style, save_fig

apply_style()
sns.set_palette('deep')

FIG_DIR = os.path.join(REPO_ROOT, 'figures')
TABLE_DIR = os.path.join(REPO_ROOT, 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

raw = fetch_raw()
print('raw shape:', raw.shape)
clean_df = clean(raw)
print('clean shape:', clean_df.shape)
"""


def build_eda_notebook():
    cells = []
    cells.append(_md(
        "# 01 — Exploratory data analysis\n"
        "\n"
        "**Dataset.** UCI ML Repository #264 — *EEG Eye State*. 14,980 samples,\n"
        "14 EEG channels (Emotiv EPOC headset, sampled at ~128 Hz over ~117 s on\n"
        "a single subject) plus one binary label, `eyeDetection` (0 = open,\n"
        "1 = closed).\n"
        "\n"
        "**Research question.** Can we classify the per-sample eye state from the\n"
        "14-channel voltage vector at that instant? The supervised models will\n"
        "land in `02_modeling.ipynb` — here we focus on understanding the data\n"
        "and on motivating a temporally honest evaluation scheme.\n"
        "\n"
        "**Methods palette.** This project is constrained to the COGS 109 Spring\n"
        "2026 study guide. In this notebook the two modeling methods we *do* use\n"
        "(both as EDA tools, not as classifiers) are **hierarchical clustering**\n"
        "(figure 06) and **K-means k=2** (figure 07). The supervised classifiers\n"
        "(LDA, KNN, PCA→LDA, PCR-as-classifier) are coming in\n"
        "`02_modeling.ipynb`.\n"
    ))
    cells.append(_code(EDA_CODE_SETUP.strip()))

    # 1: class balance
    cells.append(_md(
        "## Figure 01 — Class balance\n"
        "\n"
        "About 55% of samples are eyes-open and 45% eyes-closed. The dataset is\n"
        "near-balanced, so accuracy is a meaningful headline metric (we will\n"
        "still report sensitivity and specificity in the modeling notebook).\n"
    ))
    cells.append(_code(
        "counts = raw[LABEL_COL].value_counts().sort_index()\n"
        "labels = ['eyes open (0)', 'eyes closed (1)']\n"
        "fig, ax = plt.subplots(figsize=(5.5, 3.5))\n"
        "bars = ax.bar(labels, counts.values, color=['#4c72b0', '#dd8452'])\n"
        "for b, c in zip(bars, counts.values):\n"
        "    pct = 100 * c / counts.sum()\n"
        "    ax.text(b.get_x() + b.get_width()/2, c + 80, f'{c}\\n({pct:.1f}%)',\n"
        "            ha='center', fontsize=10)\n"
        "ax.set_ylabel('count')\n"
        "ax.set_title('Class balance — UCI EEG Eye State (n=14,980)')\n"
        "ax.set_ylim(0, counts.max() * 1.18)\n"
        "save_fig(fig, '01_class_balance.png'); plt.show()"
    ))

    # 2: channel boxplots (raw, pre-clean)
    cells.append(_md(
        "## Figure 02 — Per-channel boxplots (raw, pre-clean)\n"
        "\n"
        "The Emotiv recording has a handful of giant sensor jumps in the raw\n"
        "stream. The whiskers below make those outliers visually obvious — they\n"
        "are what our `|z|>4` cleaning step removes.\n"
    ))
    cells.append(_code(
        "fig, ax = plt.subplots(figsize=(11, 4))\n"
        "raw[list(CHANNELS)].boxplot(ax=ax, showfliers=True)\n"
        "ax.set_ylabel('voltage (μV)')\n"
        "ax.set_title('Per-channel boxplots (raw data)')\n"
        "plt.xticks(rotation=0)\n"
        "save_fig(fig, '02_channel_boxplots.png'); plt.show()"
    ))

    # 3: histograms raw + clean
    cells.append(_md(
        "## Figure 03 — Per-channel histograms (raw vs cleaned)\n"
        "\n"
        "The first grid shows the heavy tails of the raw recording on a log\n"
        "y-axis; the second shows the same channels after dropping the four\n"
        "rows with any `|z|>4`. The tails collapse to roughly Gaussian shapes,\n"
        "confirming the outliers are a tiny handful of extreme samples and not\n"
        "a structural property of the signal.\n"
    ))
    cells.append(_code(
        "def _channel_hist_grid(df, title, fname, log_y=True):\n"
        "    fig, axes = plt.subplots(2, 7, figsize=(15, 5), sharex=False, sharey=False)\n"
        "    for ch, ax in zip(CHANNELS, axes.flatten()):\n"
        "        ax.hist(df[ch], bins=60, color='#4c72b0', alpha=0.85)\n"
        "        if log_y:\n"
        "            ax.set_yscale('log')\n"
        "        ax.set_title(ch, fontsize=10)\n"
        "        ax.tick_params(labelsize=8)\n"
        "    fig.suptitle(title)\n"
        "    fig.tight_layout()\n"
        "    save_fig(fig, fname); plt.show()\n"
        "\n"
        "_channel_hist_grid(raw, 'Per-channel histograms (raw, log y)', '03_channel_histograms_raw.png')\n"
        "_channel_hist_grid(clean_df, 'Per-channel histograms (after |z|>4 drop, log y)', '03_channel_histograms_clean.png')"
    ))

    # 4: time series with label strip + 4b folds overlay
    cells.append(_md(
        "## Figure 04 — Stacked 14-channel time series with label strip\n"
        "\n"
        "This is the headline figure. All 14 channels are shown on the same\n"
        "time axis (z-scored and offset for readability) and the bottom strip\n"
        "indicates whether the eyes were open (blue) or closed (orange) at each\n"
        "sample. Visual inspection already suggests that the channel statistics\n"
        "drift over multi-second windows, which is exactly what the supervised\n"
        "model will need to exploit.\n"
    ))
    cells.append(_code(
        "scaler = fit_scaler(clean_df)\n"
        "clean_z = apply_scaler(clean_df, scaler)\n"
        "\n"
        "def _plot_stacked_timeseries(df_z, fold_boundaries=None, fname='04_timeseries_with_label.png',\n"
        "                            title='Stacked 14-channel EEG with eye-state label strip'):\n"
        "    fig, (ax_sig, ax_lab) = plt.subplots(2, 1, figsize=(14, 7),\n"
        "                                         gridspec_kw={'height_ratios': [10, 1]}, sharex=True)\n"
        "    n = len(df_z)\n"
        "    t = np.arange(n)\n"
        "    offset = 5.0\n"
        "    for i, ch in enumerate(CHANNELS):\n"
        "        ax_sig.plot(t, df_z[ch].values + i * offset, lw=0.4, color='#333')\n"
        "        ax_sig.text(-30, i * offset, ch, ha='right', va='center', fontsize=8)\n"
        "    ax_sig.set_yticks([])\n"
        "    ax_sig.set_ylabel('channel (offset, z)')\n"
        "    ax_sig.set_title(title)\n"
        "    ax_sig.set_xlim(0, n)\n"
        "    if fold_boundaries is not None:\n"
        "        for b in fold_boundaries:\n"
        "            ax_sig.axvline(b, color='#c44e52', lw=0.8, alpha=0.8)\n"
        "    lab = df_z[LABEL_COL].values.reshape(1, -1)\n"
        "    ax_lab.imshow(lab, aspect='auto', cmap='coolwarm', interpolation='nearest', extent=[0, n, 0, 1])\n"
        "    ax_lab.set_yticks([])\n"
        "    ax_lab.set_xlabel('sample index')\n"
        "    ax_lab.set_ylabel('label', fontsize=9)\n"
        "    fig.tight_layout()\n"
        "    save_fig(fig, fname); plt.show()\n"
        "\n"
        "_plot_stacked_timeseries(clean_z)"
    ))
    cells.append(_md(
        "## Figure 04b — Same series with 5-fold time-block CV boundaries\n"
        "\n"
        "Vertical red lines mark the boundaries of the contiguous 5-fold blocked\n"
        "CV scheme over the chronological training partition. Each block spans\n"
        "many open/closed transitions, so each fold's test set genuinely sees\n"
        "label dynamics that the train set didn't see at adjacent indices.\n"
    ))
    cells.append(_code(
        "train_chron, _ = split_chronological(clean_df, test_frac=0.20, seam_gap=64)\n"
        "blocked = blocked_kfold_indices(len(train_chron), n_splits=5)\n"
        "boundaries = [int(te[0]) for _, te in blocked][1:] + [int(blocked[-1][1][-1]) + 1]\n"
        "train_chron_z = apply_scaler(train_chron, scaler)\n"
        "_plot_stacked_timeseries(train_chron_z, fold_boundaries=boundaries,\n"
        "                        fname='04b_timeseries_with_folds.png',\n"
        "                        title='Chronological train partition with 5-fold time-block CV boundaries')"
    ))

    # 5: corr heatmap
    cells.append(_md(
        "## Figure 05 — Channel-by-channel correlation heatmap (cleaned)\n"
        "\n"
        "Pearson correlation across the 14 channels on the outlier-cleaned data.\n"
        "Frontal channels (AF3/F7/F3/F4/F8/AF4) cluster strongly with each other,\n"
        "as do the temporal pair T7/T8 and occipital pair O1/O2. This is the\n"
        "structure we re-discover with hierarchical clustering in figure 06.\n"
    ))
    cells.append(_code(
        "corr = clean_df[list(CHANNELS)].corr()\n"
        "fig, ax = plt.subplots(figsize=(7, 6))\n"
        "sns.heatmap(corr, annot=False, cmap='RdBu_r', vmin=-1, vmax=1, square=True, ax=ax, cbar_kws={'shrink': 0.75})\n"
        "ax.set_title('14-channel Pearson correlation (cleaned)')\n"
        "save_fig(fig, '05_channel_correlation_heatmap.png'); plt.show()"
    ))

    # 6: dendrograms
    cells.append(_md(
        "## Figure 06 — Hierarchical clustering dendrograms (study-guide method)\n"
        "\n"
        "Distance matrix is `1 − |corr|` over the 14 channels (so high-correlation\n"
        "channels are close). We render single-linkage and complete-linkage\n"
        "dendrograms side-by-side. Both linkages discover the frontal block as\n"
        "one tight cluster; complete linkage additionally pulls O1/O2 and T7/T8\n"
        "into clean pairs.\n"
    ))
    cells.append(_code(
        "from scipy.cluster.hierarchy import linkage, dendrogram\n"
        "from scipy.spatial.distance import squareform\n"
        "\n"
        "dist = 1.0 - corr.abs()\n"
        "np.fill_diagonal(dist.values, 0.0)\n"
        "condensed = squareform(dist.values, checks=False)\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))\n"
        "for ax, method in zip(axes, ['single', 'complete']):\n"
        "    Z = linkage(condensed, method=method)\n"
        "    dendrogram(Z, labels=list(CHANNELS), leaf_rotation=0, ax=ax, color_threshold=0.6)\n"
        "    ax.set_title(f'{method}-linkage dendrogram (1 − |corr|)')\n"
        "    ax.set_ylabel('distance')\n"
        "fig.tight_layout()\n"
        "save_fig(fig, '06_channel_dendrogram.png'); plt.show()"
    ))

    # 7: k-means k=2 vs label
    cells.append(_md(
        "## Figure 07 — K-means k=2 vs ground-truth label (study-guide method)\n"
        "\n"
        "K-means with k=2 is fit on the z-scored 14-channel sample vectors *with\n"
        "no access to the label*. The strip plot beneath shows the predicted\n"
        "cluster assignment (aligned to the label via majority-vote) and the\n"
        "true label. We report the unsupervised agreement rate as a sanity\n"
        "check — it should be modestly above chance (the EEG signal is noisy\n"
        "and instantaneous-sample classification is hard for an unsupervised\n"
        "method) but well below what the supervised classifiers will reach.\n"
    ))
    cells.append(_code(
        "from sklearn.cluster import KMeans\n"
        "X = clean_z[list(CHANNELS)].values\n"
        "y = clean_z[LABEL_COL].values.astype(int)\n"
        "km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)\n"
        "pred = km.labels_\n"
        "# align cluster ids to label majority\n"
        "agree = max((pred == y).mean(), (pred != y).mean())\n"
        "if (pred == y).mean() < (pred != y).mean():\n"
        "    pred = 1 - pred\n"
        "fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 2.6), sharex=True)\n"
        "ax_top.imshow(y.reshape(1, -1), aspect='auto', cmap='coolwarm', interpolation='nearest')\n"
        "ax_top.set_yticks([]); ax_top.set_ylabel('true', fontsize=9)\n"
        "ax_top.set_title(f'K-means k=2 vs true eye-state label — unsupervised agreement = {agree*100:.1f}%')\n"
        "ax_bot.imshow(pred.reshape(1, -1), aspect='auto', cmap='coolwarm', interpolation='nearest')\n"
        "ax_bot.set_yticks([]); ax_bot.set_ylabel('K-means', fontsize=9)\n"
        "ax_bot.set_xlabel('sample index')\n"
        "fig.tight_layout()\n"
        "save_fig(fig, '07_kmeans_vs_label.png'); plt.show()\n"
        "print(f'unsupervised K-means k=2 agreement: {agree*100:.2f}%')"
    ))

    # 8: autocorr — THE figure
    cells.append(_md(
        "## Figure 08 — Label autocorrelation by lag (motivates blocked CV)\n"
        "\n"
        "This is the empirical justification for why we run **two** CV schemes\n"
        "in Phase B. Lag-1 autocorrelation of the binary `eyeDetection` label\n"
        "is r ≈ 0.997. Even at lag 256 (≈ 2 seconds at 128 Hz), r is still\n"
        "around 0.47. Adjacent samples are nearly identical labels, so a\n"
        "shuffled k-fold CV will place near-duplicate train/test rows side by\n"
        "side and report inflated accuracy. A time-blocked k-fold puts entire\n"
        "open/closed bouts on one side of the train/test boundary and gives an\n"
        "honest estimate of how the model would generalise to *new* time.\n"
    ))
    cells.append(_code(
        "def label_autocorr(y, lags):\n"
        "    y = np.asarray(y, dtype=float)\n"
        "    y = y - y.mean()\n"
        "    var = (y * y).mean()\n"
        "    out = []\n"
        "    for k in lags:\n"
        "        if k == 0:\n"
        "            out.append(1.0)\n"
        "            continue\n"
        "        out.append(float((y[:-k] * y[k:]).mean() / var))\n"
        "    return out\n"
        "\n"
        "lags = [1, 5, 10, 32, 64, 128, 256, 512, 1024]\n"
        "rs = label_autocorr(clean_df[LABEL_COL].values, lags)\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "ax.plot(lags, rs, marker='o', color='#c44e52')\n"
        "for lag_val, acorr_val in zip(lags, rs):\n"
        "    ax.annotate(f'{acorr_val:.2f}', xy=(lag_val, acorr_val), xytext=(5, 6), textcoords='offset points', fontsize=9)\n"
        "ax.set_xscale('log')\n"
        "ax.set_xlabel('lag (samples; ~128 Hz)')\n"
        "ax.set_ylabel('label autocorrelation r')\n"
        "ax.axhline(0, color='#888', lw=0.6)\n"
        "ax.set_title('eyeDetection autocorrelation — motivation for blocked CV')\n"
        "save_fig(fig, '08_label_autocorrelation.png'); plt.show()\n"
        "for k, r in zip(lags, rs):\n"
        "    print(f'  lag {k:5d}  r = {r: .3f}')"
    ))

    # 9: PCA scatter
    cells.append(_md(
        "## Figure 09 — PCA scatter (PC1 vs PC2, colored by label)\n"
        "\n"
        "Two principal components of the z-scored 14-channel matrix. The two\n"
        "classes overlap substantially in the first two PCs, which is expected\n"
        "given that the class signal is in the *covariance structure across\n"
        "channels and time*, not in a single dominant linear direction. PCA\n"
        "(and PCR-as-classifier) is one of the study-guide methods Child B will\n"
        "still use, with > 2 components.\n"
    ))
    cells.append(_code(
        "from sklearn.decomposition import PCA\n"
        "p = PCA(n_components=2, random_state=42).fit(X)\n"
        "Z = p.transform(X)\n"
        "fig, ax = plt.subplots(figsize=(6.5, 5))\n"
        "for cls, c in [(0, '#4c72b0'), (1, '#dd8452')]:\n"
        "    mask = (y == cls)\n"
        "    ax.scatter(Z[mask, 0], Z[mask, 1], s=3, alpha=0.35, label=('open' if cls == 0 else 'closed'), c=c)\n"
        "ax.set_xlabel(f'PC1 ({p.explained_variance_ratio_[0]*100:.1f}% var)')\n"
        "ax.set_ylabel(f'PC2 ({p.explained_variance_ratio_[1]*100:.1f}% var)')\n"
        "ax.legend(markerscale=3, loc='best')\n"
        "ax.set_title('PCA on z-scored 14-channel EEG, colored by eye-state')\n"
        "save_fig(fig, '09_pca_scatter.png'); plt.show()"
    ))

    # tables
    cells.append(_md(
        "## Tables — summary statistics and outlier ablation\n"
        "\n"
        "* `tables/01_summary_stats.csv` — per-channel min, max, mean, median,\n"
        "  std, IQR, both for the raw recording and for the cleaned version.\n"
        "* `tables/02_outlier_ablation.csv` — compares three outlier-handling\n"
        "  strategies (drop, winsorize at z=4, clip at z=4). We report the\n"
        "  post-cleaning row count, the aggregate (per-channel mean of)\n"
        "  absolute shift in mean and std versus the raw stats, and our\n"
        "  recommended choice with a one-line justification.\n"
    ))
    cells.append(_code(
        "def _summary(df):\n"
        "    rows = []\n"
        "    for ch in CHANNELS:\n"
        "        s = df[ch].astype(float)\n"
        "        rows.append({\n"
        "            'channel': ch,\n"
        "            'min': s.min(), 'max': s.max(),\n"
        "            'mean': s.mean(), 'median': s.median(),\n"
        "            'std': s.std(ddof=0),\n"
        "            'iqr': s.quantile(0.75) - s.quantile(0.25),\n"
        "        })\n"
        "    return pd.DataFrame(rows)\n"
        "\n"
        "raw_stats = _summary(raw).add_prefix('raw_').rename(columns={'raw_channel': 'channel'})\n"
        "clean_stats = _summary(clean_df).add_prefix('clean_').rename(columns={'clean_channel': 'channel'})\n"
        "summary = raw_stats.merge(clean_stats, on='channel')\n"
        "summary.to_csv(os.path.join(TABLE_DIR, '01_summary_stats.csv'), index=False)\n"
        "summary.head()"
    ))
    cells.append(_code(
        "# Outlier ablation: drop vs winsorize vs clip at |z|=4\n"
        "ch_data = raw[list(CHANNELS)].astype(float)\n"
        "mu_raw = ch_data.mean(axis=0)\n"
        "sd_raw = ch_data.std(axis=0, ddof=0)\n"
        "z = (ch_data - mu_raw) / sd_raw\n"
        "mask_keep = (z.abs() <= 4).all(axis=1)\n"
        "\n"
        "def _ablate(strategy):\n"
        "    if strategy == 'drop':\n"
        "        out = raw.loc[mask_keep, list(CHANNELS)].astype(float).copy()\n"
        "    else:\n"
        "        out = ch_data.copy()\n"
        "        for c in CHANNELS:\n"
        "            lo, hi = mu_raw[c] - 4 * sd_raw[c], mu_raw[c] + 4 * sd_raw[c]\n"
        "            if strategy == 'winsorize':\n"
        "                lo_v = out[c].quantile(0.005); hi_v = out[c].quantile(0.995)\n"
        "                out[c] = out[c].clip(lo_v, hi_v)\n"
        "            elif strategy == 'clip':\n"
        "                out[c] = out[c].clip(lo, hi)\n"
        "            else:\n"
        "                raise ValueError(strategy)\n"
        "    new_mu = out.mean(axis=0); new_sd = out.std(axis=0, ddof=0)\n"
        "    mean_shift = (new_mu - mu_raw).abs().mean()\n"
        "    std_shift = (new_sd - sd_raw).abs().mean()\n"
        "    return {\n"
        "        'strategy': strategy,\n"
        "        'n_rows_kept': int(len(out)),\n"
        "        'avg_abs_mean_shift_uV': float(mean_shift),\n"
        "        'avg_abs_std_shift_uV': float(std_shift),\n"
        "    }\n"
        "\n"
        "rows = [_ablate(s) for s in ['drop', 'winsorize', 'clip']]\n"
        "ablation = pd.DataFrame(rows)\n"
        "ablation['recommended'] = ablation['strategy'].eq('drop')\n"
        "ablation['justification'] = [\n"
        "    'drop is preferred: only 4 rows (0.027%) are removed and the per-channel mean/std stay closest to the bulk-data values.',\n"
        "    'winsorize at empirical 0.5/99.5% deforms many in-distribution samples to make outliers benign; over-corrects.',\n"
        "    'clip at z=4 also alters non-outlier rows because z is computed under the outlier-inflated std, biasing the threshold.',\n"
        "]\n"
        "ablation.to_csv(os.path.join(TABLE_DIR, '02_outlier_ablation.csv'), index=False)\n"
        "ablation"
    ))

    cells.append(_md(
        "## EDA findings → modeling implications\n"
        "\n"
        "* **Temporal leakage matters.** With lag-1 label autocorrelation\n"
        "  r ≈ 0.997 (figure 08), the project must contrast shuffled and\n"
        "  blocked k-fold CV. Both fold-index files are frozen in\n"
        "  `data/processed/cv_folds.json` so `02_modeling.ipynb` can compare\n"
        "  them honestly.\n"
        "* **Hierarchical clustering and K-means k=2 are the two EDA-side\n"
        "  models** (figures 06 and 07). Both are explicitly in the COGS 109\n"
        "  methods palette. The supervised classifiers — LDA, KNN, PCA→LDA,\n"
        "  and PCR-as-classifier — will be evaluated in `02_modeling.ipynb`\n"
        "  on top of the same frozen splits and z-score scalers produced by\n"
        "  this notebook and `scripts/preprocess.py`.\n"
    ))

    nb = _nb(cells)
    nbf.write(nb, os.path.join(NB_DIR, "01_eda.ipynb"))
    print("wrote notebooks/01_eda.ipynb")


def main():
    build_fetch_notebook()
    build_eda_notebook()


if __name__ == "__main__":
    main()
