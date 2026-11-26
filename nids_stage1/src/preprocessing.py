"""
preprocessing.py
=================
Stage 1 - Component 2: Cleaning & Preprocessing

Takes the raw, loaded DataFrame (from data_loader.py) and produces a
model-ready numeric matrix.

Steps performed (in order):
    1. Drop exact duplicate rows
    2. Replace inf/-inf with NaN, then drop or impute
    3. Drop columns that are constant (zero variance -> no information)
    4. Encode categorical columns (one-hot for low-cardinality, label
       encoding for high-cardinality e.g. IP-like fields)
    5. Separate features (X) from labels (y_binary, y_multiclass)
    6. THREE-WAY stratified split: train / validation / test
    7. Feature scaling (StandardScaler, fit on train only to avoid leakage)

Why three-way, not train/test?
-------------------------------
Downstream stages need a set that is (a) NEVER used for training or
feature selection, but (b) kept at the REAL, untouched class distribution
(unlike the training set, which Stage 1 also SMOTE-resamples) so that
decision-threshold tuning (Stage 2) and short-circuit-threshold tuning
(Stage 3) are calibrated against realistic traffic, not an artificially
balanced 50/50 mix. The TEST set stays completely untouched by any tuning
decision and is used exactly once, for final reporting.

    X_train -> feature selection, SMOTE resampling, model fitting
    X_val   -> decision-threshold tuning, hyperparameter/model selection
               (real distribution, never resampled)
    X_test  -> final evaluation ONLY, touched exactly once

Each step logs what it did — this becomes your "Data Preprocessing" report
section almost verbatim.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


@dataclass
class PreprocessResult:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train_bin: pd.Series
    y_val_bin: pd.Series
    y_test_bin: pd.Series
    y_train_multi: pd.Series
    y_val_multi: pd.Series
    y_test_multi: pd.Series
    feature_names: list
    scaler: StandardScaler
    log: list = field(default_factory=list)


def _log(log_list, msg):
    log_list.append(msg)
    print(msg)


def clean_and_split(
    df: pd.DataFrame,
    label_col: str = "label",
    multiclass_col: str = "attack_type",
    drop_cols: list = None,
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 42,
    max_onehot_cardinality: int = 15,
) -> PreprocessResult:
    """
    Full cleaning + THREE-WAY train/val/test split pipeline.

    Parameters
    ----------
    df : raw unified DataFrame from data_loader.py
    label_col : binary label column (0/1)
    multiclass_col : attack-type string column
    drop_cols : any extra identifier columns to drop (e.g. 'source_dataset',
                'flow_id', 'src_ip', 'timestamp' — high-cardinality / leaky
                fields that shouldn't be fed to the model as-is)
    test_size : fraction of the FULL dataset held out for final testing
    val_size : fraction of the FULL dataset held out for validation
               (threshold tuning, model selection) — kept at the REAL
               class distribution, never resampled
    random_state : for reproducibility
    max_onehot_cardinality : categorical cols with <= this many unique
                              values get one-hot encoded; above this,
                              label-encoded instead (avoids exploding
                              dimensionality on IP-like fields)

    Returns
    -------
    PreprocessResult dataclass with train/val/test splits. Only X_train
    should ever be fed to SMOTE/resampling — X_val and X_test must stay
    at the real, untouched class distribution.
    """
    log = []
    n_start = len(df)

    default_drops = ["source_dataset", "flow_id", "timestamp", "src_ip", "dst_ip", "srcip", "dstip"]
    drop_cols = list(set((drop_cols or []) + default_drops))
    existing_drops = [c for c in drop_cols if c in df.columns]
    if existing_drops:
        df = df.drop(columns=existing_drops)
        _log(log, f"[1/7] Dropped identifier/leaky columns: {existing_drops}")

    # 1. Drop exact duplicate rows
    dup_count = df.duplicated().sum()
    df = df.drop_duplicates()
    _log(log, f"[2/7] Dropped {dup_count} duplicate rows ({n_start} -> {len(df)} rows)")

    # 2. Replace inf/-inf with NaN, then handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != label_col]
    n_inf = np.isinf(df[numeric_cols]).sum().sum()
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    n_missing_before = df[numeric_cols].isna().sum().sum()
    # Impute missing numeric values with the column median (robust to outliers,
    # standard practice for flow-based network features).
    for c in numeric_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    _log(
        log,
        f"[3/7] Replaced {int(n_inf)} inf values and imputed {int(n_missing_before)} "
        f"missing numeric values with column medians",
    )

    # 3. Drop zero-variance (constant) columns — no information for the model
    variances = df[numeric_cols].var()
    constant_cols = variances[variances == 0].index.tolist()
    if constant_cols:
        df = df.drop(columns=constant_cols)
        _log(log, f"[4/7] Dropped {len(constant_cols)} zero-variance columns: {constant_cols}")

    # 4. Encode categorical columns (everything except the two label columns)
    categorical_cols = [
        c for c in df.select_dtypes(include=["object"]).columns
        if c not in (label_col, multiclass_col)
    ]
    onehot_cols, label_enc_cols = [], []
    for c in categorical_cols:
        if df[c].nunique() <= max_onehot_cardinality:
            onehot_cols.append(c)
        else:
            label_enc_cols.append(c)

    if onehot_cols:
        df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)
    for c in label_enc_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
    _log(
        log,
        f"[5/7] Encoded categoricals -> one-hot: {onehot_cols or 'none'}, "
        f"label-encoded (high cardinality): {label_enc_cols or 'none'}",
    )

    # 5. Separate features / labels
    y_bin = df[label_col].astype(int)
    y_multi = df[multiclass_col].astype(str) if multiclass_col in df.columns else pd.Series(
        np.where(y_bin == 1, "Attack", "Benign"), index=df.index
    )
    X = df.drop(columns=[c for c in [label_col, multiclass_col] if c in df.columns])
    # keep only numeric columns from here on (should be all of them by now)
    X = X.select_dtypes(include=[np.number])
    _log(log, f"[6/7] Final feature matrix shape: {X.shape}")

    # 6. THREE-WAY stratified split: train / validation / test.
    #    Step A: carve off the test set first (never touched again until final eval).
    #    Step B: carve the validation set out of what remains (val_size is expressed
    #    as a fraction of the FULL dataset, so we rescale it relative to the
    #    remaining pool here).
    X_trainval, X_test, y_trainval_bin, y_test_bin, y_trainval_multi, y_test_multi = train_test_split(
        X, y_bin, y_multi,
        test_size=test_size,
        random_state=random_state,
        stratify=y_bin,
    )
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train_bin, y_val_bin, y_train_multi, y_val_multi = train_test_split(
        X_trainval, y_trainval_bin, y_trainval_multi,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_trainval_bin,
    )

    # 7. Scale features — fit ONLY on train, apply to val/test. Val and test
    #    stay at the REAL class distribution (no resampling applied to either) —
    #    only X_train should ever be passed to SMOTE/resampling downstream.
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), columns=X_val.columns, index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    _log(
        log,
        f"[7/7] Three-way split -> train: {X_train_scaled.shape} (real distribution, "
        f"SMOTE applied downstream), val: {X_val_scaled.shape} (REAL distribution, untouched — "
        f"use for threshold tuning), test: {X_test_scaled.shape} (REAL distribution, untouched — "
        f"final evaluation only). Scaler fit on train only.",
    )
    _log(
        log,
        f"  Class balance — train: {y_train_bin.value_counts().to_dict()}, "
        f"val: {y_val_bin.value_counts().to_dict()}, test: {y_test_bin.value_counts().to_dict()}",
    )

    return PreprocessResult(
        X_train=X_train_scaled,
        X_val=X_val_scaled,
        X_test=X_test_scaled,
        y_train_bin=y_train_bin,
        y_val_bin=y_val_bin,
        y_test_bin=y_test_bin,
        y_train_multi=y_train_multi,
        y_val_multi=y_val_multi,
        y_test_multi=y_test_multi,
        feature_names=list(X_train_scaled.columns),
        scaler=scaler,
        log=log,
    )
