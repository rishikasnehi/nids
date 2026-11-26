"""
data_loader.py
================
Stage 1 - Component 1: Data Loading

Handles loading raw CIC-IDS2017 and UNSW-NB15 CSV files and unifying them
into a common schema so the rest of the pipeline (preprocessing, feature
selection, imbalance handling, model training) doesn't need to know which
dataset it's working with.

Why this is needed:
--------------------
- CIC-IDS2017 files ship with messy column names (leading/trailing spaces,
  inconsistent capitalization) and often contain Infinity / NaN values in
  flow-rate columns (e.g. "Flow Bytes/s") caused by zero-duration flows.
- UNSW-NB15 ships as two files (a "features" training/testing split) with a
  different column naming convention altogether, and has BOTH a binary
  `label` column and a multi-class `attack_cat` column.

This module normalizes both into:
    - a cleaned column-name convention (snake_case, no stray whitespace)
    - a unified binary label column: `label` (0 = benign, 1 = attack)
    - a unified multi-class label column: `attack_type` (string)
"""

import glob
import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Column-name cleaning
# ---------------------------------------------------------------------------
def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, lowercase, replace spaces/slashes with underscores."""
    new_cols = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[ /]+", "_", regex=True)
        .str.replace(r"[^0-9a-z_]", "", regex=True)
    )
    df.columns = new_cols
    return df


# ---------------------------------------------------------------------------
# CIC-IDS2017 loader
# ---------------------------------------------------------------------------
def load_cic_ids2017(path: str) -> pd.DataFrame:
    """
    Load one or more CIC-IDS2017 CSV files.

    Parameters
    ----------
    path : str
        Either a single CSV file path, or a directory containing multiple
        CIC-IDS2017 daily CSVs (e.g. Monday-WorkingHours.pcap_ISCX.csv,
        Tuesday-WorkingHours.pcap_ISCX.csv, ...). If a directory is given,
        all CSVs inside are concatenated.

    Returns
    -------
    pd.DataFrame with a unified `label` (binary) and `attack_type` (string)
    column added, raw feature columns cleaned.
    """
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        frames = []
        for f in files:
            frame = pd.read_csv(f, low_memory=False)
            f_cols = frame.select_dtypes(include=["float64"]).columns
            if len(f_cols) > 0:
                frame[f_cols] = frame[f_cols].astype(np.float32)
            i_cols = frame.select_dtypes(include=["int64"]).columns
            if len(i_cols) > 0:
                frame[i_cols] = frame[i_cols].astype(np.int32)
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
        del frames
        import gc
        gc.collect()
    else:
        df = pd.read_csv(path, low_memory=False)
        f_cols = df.select_dtypes(include=["float64"]).columns
        if len(f_cols) > 0:
            df[f_cols] = df[f_cols].astype(np.float32)
        i_cols = df.select_dtypes(include=["int64"]).columns
        if len(i_cols) > 0:
            df[i_cols] = df[i_cols].astype(np.int32)

    df = _clean_column_names(df)

    # The label column in CIC-IDS2017 is typically just "label" after cleaning,
    # with values like "BENIGN", "DDoS", "PortScan", "Bot", etc.
    if "label" not in df.columns:
        raise KeyError(
            "Expected a 'Label' column in CIC-IDS2017 data but none was found "
            f"after cleaning. Columns present: {list(df.columns)}"
        )

    df["attack_type"] = df["label"].astype(str).str.strip()
    df["label"] = (df["attack_type"].str.upper() != "BENIGN").astype(int)

    df["source_dataset"] = "cic_ids2017"
    return df


# ---------------------------------------------------------------------------
# UNSW-NB15 loader
# ---------------------------------------------------------------------------
def load_unsw_nb15(path: str) -> pd.DataFrame:
    """
    Load one or more UNSW-NB15 CSV files (e.g. UNSW_NB15_training-set.csv,
    UNSW_NB15_testing-set.csv, or the raw UNSW-NB15_1.csv..4.csv parts).

    Returns
    -------
    pd.DataFrame with unified `label` (binary) and `attack_type` (string).
    """
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        frames = [pd.read_csv(f, low_memory=False) for f in files]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path, low_memory=False)

    df = _clean_column_names(df)

    # UNSW-NB15 typically has both 'label' (0/1) and 'attack_cat' (string,
    # 'Normal' for benign rows or blank/NaN).
    if "label" not in df.columns:
        raise KeyError(
            "Expected a 'label' column in UNSW-NB15 data but none was found "
            f"after cleaning. Columns present: {list(df.columns)}"
        )

    df["label"] = df["label"].astype(int)

    if "attack_cat" in df.columns:
        df["attack_type"] = (
            df["attack_cat"].astype(str).str.strip().replace({"nan": "Normal", "": "Normal"})
        )
    else:
        df["attack_type"] = np.where(df["label"] == 1, "Attack", "Normal")

    # id column (if present) is just a row index, not a feature
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df["source_dataset"] = "unsw_nb15"
    return df


# ---------------------------------------------------------------------------
# Generic dispatcher
# ---------------------------------------------------------------------------
def load_dataset(name: str, path: str) -> pd.DataFrame:
    """
    Dispatch to the correct loader based on dataset name.

    Parameters
    ----------
    name : str
        One of {"cic_ids2017", "unsw_nb15"}
    path : str
        File or directory path to the raw CSV(s).
    """
    name = name.lower().strip()
    if name in ("cic_ids2017", "cic-ids2017", "cicids2017"):
        return load_cic_ids2017(path)
    elif name in ("unsw_nb15", "unsw-nb15", "unswnb15"):
        return load_unsw_nb15(path)
    else:
        raise ValueError(f"Unknown dataset name: {name}")


if __name__ == "__main__":
    print("data_loader.py — run generate_sample_data.py first, then pipeline.py")
