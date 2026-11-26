"""
pipeline.py
===========
Stage 1 - Full Orchestration

Runs the complete Stage 1 flow:
    load -> clean/split -> feature selection -> imbalance handling -> save

Run this directly:
    python pipeline.py --dataset cic_ids2017 --path ../data/sample_cic_ids2017.csv
    python pipeline.py --dataset unsw_nb15   --path ../data/sample_unsw_nb15.csv

Outputs (written to ../data/processed/<dataset>/):
    X_train.csv, X_val.csv, X_test.csv, y_*_bin.csv     -> after full cleaning, 3-way split
    X_train_selected.csv, X_val_selected.csv,
      X_test_selected.csv                                -> after feature selection
    X_train_resampled.csv, y_train_resampled.csv         -> after SMOTE (TRAIN ONLY —
                                                              val/test are NEVER resampled)
    feature_ranking_ig.csv, feature_ranking_rfe.csv      -> ranked feature tables
    stage1_report.txt                                    -> human-readable summary

IMPORTANT: X_val_selected / y_val_bin are kept at the REAL class
distribution (not SMOTE-balanced) specifically so Stage 2 and Stage 3 can
tune decision thresholds against realistic traffic rather than an
artificially balanced 50/50 mix. Never resample the validation or test sets.
"""

import argparse
import os
import sys
import json

sys.path.append(os.path.dirname(__file__))

from data_loader import load_dataset
from preprocessing import clean_and_split
from feature_selection import select_by_information_gain, select_by_rfe, combine_rankings, apply_selection
from imbalance import smote_oversample, compute_class_weights, summarize_class_distribution


def run_stage1(dataset_name: str, raw_path: str, top_k_features: int = 20, out_dir: str = None):
    report_lines = []

    def log(msg):
        print(msg)
        report_lines.append(str(msg))

    log(f"{'='*70}\nSTAGE 1 PIPELINE — dataset: {dataset_name}\n{'='*70}")

    # --- 1. Load ---
    log("\n>>> Step 1: Loading raw data")
    df = load_dataset(dataset_name, raw_path)
    log(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    log(f"Attack-type distribution:\n{df['attack_type'].value_counts().to_string()}")

    # --- 2. Clean + split ---
    log("\n>>> Step 2: Cleaning, encoding, splitting")
    result = clean_and_split(df)
    del df
    import gc
    gc.collect()

    for line in result.log:
        log("  " + line)

    log("\nBinary class distribution (train):")
    log(summarize_class_distribution(result.y_train_bin, "train (before resampling)").to_string())

    # --- 3. Feature selection ---
    log(f"\n>>> Step 3: Feature selection (top {top_k_features})")
    ig_ranked, ig_top = select_by_information_gain(result.X_train, result.y_train_bin, top_k=top_k_features)
    log(f"Top {top_k_features} features by Information Gain:\n{ig_ranked.head(top_k_features).to_string(index=False)}")

    rfe_ranked, rfe_top = select_by_rfe(result.X_train, result.y_train_bin, top_k=top_k_features)
    log(f"\nTop {top_k_features} features by RFE:\n{rfe_ranked.head(top_k_features).to_string(index=False)}")

    merged, agreed, union = combine_rankings(ig_ranked, rfe_ranked, top_k=top_k_features)
    log(f"\nFeatures BOTH methods agree on ({len(agreed)}): {agreed}")

    final_features = agreed if len(agreed) >= 8 else union[:top_k_features]
    log(f"\nFinal selected feature set ({len(final_features)} features): {final_features}")

    X_train_sel = result.X_train[final_features]
    X_val_sel = result.X_val[final_features]
    X_test_sel = result.X_test[final_features]

    # --- 4. Imbalance handling ---
    # CRITICAL: SMOTE is applied to X_train ONLY. X_val and X_test stay at
    # the real, untouched class distribution — they exist specifically so
    # downstream threshold tuning (Stage 2/3) is calibrated against
    # realistic traffic, not an artificially balanced training distribution.
    log("\n>>> Step 4: Class imbalance handling (SMOTE on TRAINING SET ONLY — val/test untouched)")
    X_train_resampled, y_train_resampled = smote_oversample(X_train_sel, result.y_train_bin)
    log(summarize_class_distribution(y_train_resampled, "train (after SMOTE)").to_string())
    log(summarize_class_distribution(result.y_val_bin, "validation (REAL distribution, untouched)").to_string())
    log(summarize_class_distribution(result.y_test_bin, "test (REAL distribution, untouched)").to_string())

    class_weights = compute_class_weights(result.y_train_bin)
    log(f"\nAlternative — computed class weights (for weighted-loss training instead of SMOTE): {class_weights}")

    # --- 5. Save everything ---
    out_dir = out_dir or f"../data/processed/{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)

    def _save_csv(obj, filepath):
        obj.to_csv(filepath, index=False, chunksize=50000)

    _save_csv(ig_ranked, f"{out_dir}/feature_ranking_ig.csv")
    _save_csv(rfe_ranked, f"{out_dir}/feature_ranking_rfe.csv")

    _save_csv(X_train_sel, f"{out_dir}/X_train_selected.csv")
    _save_csv(X_val_sel, f"{out_dir}/X_val_selected.csv")
    _save_csv(X_test_sel, f"{out_dir}/X_test_selected.csv")

    _save_csv(X_train_resampled, f"{out_dir}/X_train_resampled.csv")
    _save_csv(y_train_resampled, f"{out_dir}/y_train_resampled.csv")

    _save_csv(result.y_train_bin, f"{out_dir}/y_train_bin.csv")
    _save_csv(result.y_val_bin, f"{out_dir}/y_val_bin.csv")
    _save_csv(result.y_test_bin, f"{out_dir}/y_test_bin.csv")
    _save_csv(result.y_train_multi, f"{out_dir}/y_train_multi.csv")
    _save_csv(result.y_val_multi, f"{out_dir}/y_val_multi.csv")
    _save_csv(result.y_test_multi, f"{out_dir}/y_test_multi.csv")

    _save_csv(result.X_train, f"{out_dir}/X_train.csv")
    _save_csv(result.X_val, f"{out_dir}/X_val.csv")
    _save_csv(result.X_test, f"{out_dir}/X_test.csv")

    with open(f"{out_dir}/selected_features.json", "w") as f:
        json.dump(final_features, f, indent=2)

    # The scaler was fit on ALL cleaned features (before selection) — Stage 6
    # must feed it the full feature vector in this exact column order, THEN
    # subset down to `selected_features.json` afterward. Save that full
    # ordered list so live inference can reconstruct it correctly.
    with open(f"{out_dir}/full_feature_names.json", "w") as f:
        json.dump(result.feature_names, f, indent=2)

    # Persist the fitted scaler — Stage 6 (real-time inference) needs this
    # EXACT scaler to transform live traffic consistently with how the
    # model was trained. Re-fitting a new scaler on live data would use
    # different mean/std statistics than training and silently corrupt
    # every downstream prediction.
    import pickle
    with open(f"{out_dir}/scaler.pkl", "wb") as f:
        pickle.dump(result.scaler, f)
    log(f"Saved fitted StandardScaler to {out_dir}/scaler.pkl (required by Stage 6 for live inference)")

    with open(f"{out_dir}/stage1_report.txt", "w", encoding="utf-8", errors="replace") as f:
        f.write("\n".join(report_lines))

    log(f"\n>>> Stage 1 complete. All artifacts written to: {out_dir}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 1: data pipeline for NIDS project")
    parser.add_argument("--dataset", required=True, choices=["cic_ids2017", "unsw_nb15"])
    parser.add_argument("--path", required=True, help="Path to raw CSV file or directory")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top features to select")
    parser.add_argument("--out_dir", default=None, help="Output directory for processed artifacts")
    args = parser.parse_args()

    run_stage1(args.dataset, args.path, top_k_features=args.top_k, out_dir=args.out_dir)
