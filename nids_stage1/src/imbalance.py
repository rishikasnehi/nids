"""
imbalance.py
============
Stage 1 - Component 4: Class Imbalance Handling

Network intrusion datasets are heavily skewed (millions of BENIGN rows vs.
a handful of rare attack types like Heartbleed/Infiltration). This module
gives you two ways to handle it:

    1. `smote_oversample()` — a from-scratch SMOTE (Synthetic Minority
       Oversampling Technique) implementation. We wrote this ourselves
       instead of relying on the `imbalanced-learn` library so it's fully
       explainable in your viva: for each minority-class sample, find its
       k nearest minority-class neighbors and synthesize new points along
       the line segments joining them.

    2. `compute_class_weights()` — a lighter-weight alternative: instead of
       generating synthetic rows, tell the classifier to penalize mistakes
       on rare classes more heavily. Cheaper, no synthetic data risk, and
       a good ablation comparison against SMOTE in your report.

Use SMOTE for the training set only — NEVER oversample the test set, or
your evaluation metrics become meaningless (you'd be testing on synthetic
data partially derived from itself).
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_class_weight


def smote_oversample(
    X: pd.DataFrame,
    y: pd.Series,
    k_neighbors: int = 5,
    random_state: int = 42,
    sampling_strategy: str = "auto",
):
    """
    From-scratch SMOTE implementation.

    Parameters
    ----------
    X : training features (numeric, already scaled)
    y : training labels (binary or multiclass)
    k_neighbors : how many nearest same-class neighbors to interpolate between
    sampling_strategy : "auto" oversamples every minority class up to match
                         the majority class count. Alternatively pass a dict
                         {class_label: target_count} for finer control.

    Returns
    -------
    X_resampled, y_resampled : with synthetic minority samples appended
    """
    rng = np.random.RandomState(random_state)
    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)

    class_counts = pd.Series(y_arr).value_counts()
    majority_count = class_counts.max()

    if sampling_strategy == "auto":
        targets = {cls: majority_count for cls in class_counts.index}
    else:
        targets = sampling_strategy

    X_synthetic_list = []
    y_synthetic_list = []

    for cls, count in class_counts.items():
        n_needed = targets.get(cls, count) - count
        if n_needed <= 0:
            continue

        X_cls = X_arr[y_arr == cls]
        n_samples_cls = X_cls.shape[0]

        k = min(k_neighbors, n_samples_cls - 1)
        if k < 1:
            # too few samples in this class to interpolate — skip
            # (flag this in your report: this class needs raw data collection,
            # not synthetic oversampling)
            continue

        nn = NearestNeighbors(n_neighbors=k + 1).fit(X_cls)
        _, neighbor_idx = nn.kneighbors(X_cls)

        for _ in range(n_needed):
            i = rng.randint(0, n_samples_cls)
            neighbor_choice = rng.randint(1, k + 1)  # skip index 0 (self)
            j = neighbor_idx[i, neighbor_choice]

            gap = rng.rand()
            synthetic_point = X_cls[i] + gap * (X_cls[j] - X_cls[i])

            X_synthetic_list.append(synthetic_point)
            y_synthetic_list.append(cls)

    if X_synthetic_list:
        X_synthetic = np.vstack(X_synthetic_list)
        y_synthetic = np.array(y_synthetic_list)
        X_resampled = np.vstack([X_arr, X_synthetic])
        y_resampled = np.concatenate([y_arr, y_synthetic])
    else:
        X_resampled, y_resampled = X_arr, y_arr

    if isinstance(X, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled, name=y.name if hasattr(y, "name") else "label")

    return X_resampled, y_resampled


def compute_class_weights(y: pd.Series) -> dict:
    """
    Compute inverse-frequency class weights for use directly in
    RandomForestClassifier(class_weight=...) or XGBoost's sample_weight.

    Cheaper alternative to SMOTE — no synthetic rows, just re-weights the
    loss function so rare classes matter more during training.
    """
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


def summarize_class_distribution(y: pd.Series, label: str = "") -> pd.DataFrame:
    """Quick helper to print/report before-vs-after class balance."""
    counts = y.value_counts().sort_index()
    pct = (counts / counts.sum() * 100).round(2)
    summary = pd.DataFrame({"count": counts, "pct": pct})
    if label:
        print(f"\n--- Class distribution: {label} ---")
        print(summary)
    return summary
