# NIDS Project — Stage 1: Data Pipeline

This implements **Stage 1** of the project plan: data loading, cleaning,
feature selection, and class-imbalance handling for CIC-IDS2017 and
UNSW-NB15.

## 🔧 Updated after methodology review

A methodology review of the full Stages 1-3 pipeline flagged a real gap:
this stage originally only produced a train/test split, with no separate
validation set. That meant Stage 2 and Stage 3 had no clean, untouched,
realistic-distribution data to tune decision thresholds against — they
were forced to either use a default 0.5 threshold (Stage 2) or carve an
ad-hoc validation split out of the already-resampled training set
(Stage 3), neither of which is correct practice.

**Fixed**: `clean_and_split()` now produces a proper **three-way split**:

| Split | Distribution | Used for |
|---|---|---|
| `X_train` / `y_train` | SMOTE-resampled (50/50) | Feature selection, model fitting |
| `X_val` / `y_val` | **Real, untouched** (e.g. ~80/20) | Decision-threshold tuning, model selection — **never** used for training |
| `X_test` / `y_test` | **Real, untouched** (e.g. ~80/20) | Final evaluation, used exactly once |

New output files: `X_val.csv`, `X_val_selected.csv`, `y_val_bin.csv`,
`y_val_multi.csv`. Everything else (feature selection learned from train
only, SMOTE applied to train only, scaler fit on train only) was already
correct and is unchanged.

---

## What's included

```
nids_stage1/
├── src/
│   ├── data_loader.py         # Loads + unifies CIC-IDS2017 / UNSW-NB15 raw CSVs
│   ├── preprocessing.py       # Cleaning, encoding, train/test split, scaling
│   ├── feature_selection.py   # Information Gain + RFE feature ranking
│   ├── imbalance.py           # From-scratch SMOTE + class-weight alternative
│   ├── pipeline.py            # Orchestrates the full Stage 1 flow end-to-end
│   └── generate_sample_data.py# Creates SYNTHETIC test data (for pipeline testing only)
├── data/
│   ├── sample_cic_ids2017.csv # synthetic — for testing only
│   ├── sample_unsw_nb15.csv   # synthetic — for testing only
│   └── processed/             # pipeline outputs land here
└── README.md
```

## ⚠️ Important: this currently runs on SYNTHETIC data

I don't have network access to download the real CIC-IDS2017 / UNSW-NB15
datasets, so I generated small synthetic CSVs that **mimic their column
structure** to prove the pipeline works end-to-end (inf/NaN handling,
categorical encoding, feature selection, SMOTE all verified working on both
schemas). **You must swap in the real datasets before training any model you
report results on.**

## How to plug in the real datasets

1. Download the real data:
   - **CIC-IDS2017**: https://www.unb.ca/cic/datasets/ids-2017.html (the
     "MachineLearningCSV" version — 8 daily CSV files)
   - **UNSW-NB15**: https://research.unsw.edu.au/projects/unsw-nb15-dataset
     (use `UNSW_NB15_training-set.csv` + `UNSW_NB15_testing-set.csv`, or the
     4 raw part files)

2. Put them in `data/raw/cic_ids2017/` and `data/raw/unsw_nb15/` respectively
   (a folder with multiple CSVs is fine — the loader concatenates them
   automatically).

3. Run:
   ```bash
   cd src
   python pipeline.py --dataset cic_ids2017 --path ../data/raw/cic_ids2017 --top_k 20
   python pipeline.py --dataset unsw_nb15   --path ../data/raw/unsw_nb15   --top_k 20
   ```

4. Check `data/processed/<dataset>/stage1_report.txt` — this is your
   ready-to-paste "Data Preprocessing" section for the project report. It
   logs every cleaning decision (duplicates dropped, inf/NaN handled,
   constant columns removed, features selected, class balance before/after).

## What each script does, in the order it runs

| Step | Script | What it does |
|---|---|---|
| 1 | `data_loader.py` | Loads raw CSV(s), cleans column names, unifies both datasets' very different label conventions into a common `label` (binary) + `attack_type` (string) schema |
| 2 | `preprocessing.py` | Drops duplicates/leaky ID columns, replaces inf with NaN, imputes missing values with column median, drops zero-variance columns, one-hot/label-encodes categoricals, does a **stratified** train/test split, scales features (fit on train only — no leakage) |
| 3 | `feature_selection.py` | Ranks features by **Information Gain** (mutual information) and by **RFE** (Random Forest-driven recursive elimination), then keeps features both methods agree matter |
| 4 | `imbalance.py` | Applies **SMOTE** (implemented from scratch — explainable in your viva, no black-box library) to the **training set only**, plus computes class weights as a lighter alternative you can compare against |

## Design decisions worth knowing for your report/viva

- **Why median imputation, not mean?** Network flow features (byte counts,
  durations) are heavily right-skewed with outliers; median is more robust.
- **Why scale AFTER the train/test split?** Fitting the scaler on the full
  dataset before splitting leaks test-set statistics into training —
  a common mistake that inflates reported accuracy. We fit `StandardScaler`
  on train only, then apply it to test.
- **Why SMOTE only on the training set?** Oversampling before splitting (or
  oversampling the test set) means synthetic points derived from a sample
  could end up in both train and test — an evaluation you can't trust. Real
  attacks in the test set stay real, untouched.
- **Why both Information Gain AND RFE?** They rank features by different
  principles (IG: statistical dependency; RFE: model-driven elimination).
  Features both agree on are more defensible picks than either alone — this
  overlap-based selection itself becomes a small, reportable result.
- **Why hand-rolled SMOTE instead of `imbalanced-learn`?** No network
  access in this environment to install it, but it's also a better choice
  for your viva: you can explain exactly how synthetic points are generated
  (k-NN interpolation between minority-class samples) rather than citing a
  library you can't fully explain if asked.

## 🔧 Updated for Stage 6 (real-time inference)

Added persistence of two artifacts Stage 6 needs to process live traffic
consistently with how the model was trained:
- `scaler.pkl` — the fitted `StandardScaler` (fit on train only). Live
  traffic MUST be transformed with this exact scaler, not a freshly-fit
  one, or predictions silently corrupt.
- `full_feature_names.json` — the exact column order the scaler expects
  (before feature selection narrows it down to `selected_features.json`).

---

## Next steps (Stage 2 onward)

Once you've run this on the real datasets, `data/processed/<dataset>/` gives
you everything Stage 2 (core model training) needs:
- `X_train_resampled.csv` / `y_train_resampled.csv` — SMOTE-balanced training set
- `X_val_selected.csv` / `y_val_bin.csv` — untouched, real-distribution
  validation set for decision-threshold tuning (Stage 2/3)
- `X_test_selected.csv` / `y_test_bin.csv` — untouched, real-distribution
  test set for final, one-time evaluation
- `selected_features.json` — the final feature list, so Stage 6 (live
  capture pipeline) knows exactly which CICFlowMeter columns to extract
