"""
feature_selection.py
=====================
Stage 1 - Component 3: Feature Importance Ranking & Selection

Implements the two methods called out in your base-paper brief:
    - Information Gain (mutual information between each feature and the label)
    - Recursive Feature Elimination (RFE), using a quick RandomForest as the
      estimator whose feature_importances_ drive elimination

Both methods return a ranked list of features plus a trimmed DataFrame
containing only the top-k selected features. Using both and comparing
overlap is a nice, easy "ablation" result for your report.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def select_by_information_gain(X: pd.DataFrame, y: pd.Series, top_k: int = 20, random_state: int = 42, max_samples: int = 100000):
    """
    Rank features by mutual information (Information Gain) with the label.

    Returns
    -------
    ranked_features : pd.DataFrame with columns ['feature', 'ig_score'],
                       sorted descending by score
    top_features : list of the top_k feature names
    """
    if len(X) > max_samples:
        X_sub, _, y_sub, _ = train_test_split(
            X, y, train_size=max_samples, stratify=y, random_state=random_state
        )
    else:
        X_sub, y_sub = X, y

    scores = mutual_info_classif(X_sub, y_sub, random_state=random_state, discrete_features=False)
    ranked = (
        pd.DataFrame({"feature": X.columns, "ig_score": scores})
        .sort_values("ig_score", ascending=False)
        .reset_index(drop=True)
    )
    top_features = ranked["feature"].head(top_k).tolist()
    return ranked, top_features


def select_by_rfe(X: pd.DataFrame, y: pd.Series, top_k: int = 20, random_state: int = 42, n_estimators: int = 100, max_samples: int = 100000):
    """
    Recursive Feature Elimination using a RandomForest estimator.

    Note: RFE is more expensive than mutual information (it retrains the
    estimator at each elimination step), so for very large datasets consider
    running it on a stratified sample first.

    Returns
    -------
    ranked_features : pd.DataFrame with columns ['feature', 'rfe_rank']
                       (rank 1 = most important / selected first)
    top_features : list of the top_k feature names (rank 1..top_k)
    """
    if len(X) > max_samples:
        X_sub, _, y_sub, _ = train_test_split(
            X, y, train_size=max_samples, stratify=y, random_state=random_state
        )
    else:
        X_sub, y_sub = X, y

    estimator = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    selector = RFE(estimator, n_features_to_select=top_k, step=0.1)
    selector.fit(X_sub, y_sub)

    ranked = (
        pd.DataFrame({"feature": X.columns, "rfe_rank": selector.ranking_})
        .sort_values("rfe_rank", ascending=True)
        .reset_index(drop=True)
    )
    top_features = ranked.loc[ranked["rfe_rank"] == 1, "feature"].tolist()
    return ranked, top_features


def combine_rankings(ig_ranked: pd.DataFrame, rfe_ranked: pd.DataFrame, top_k: int = 20):
    """
    Merge IG and RFE rankings into one table and return the union + the
    intersection of their top-k picks. Features both methods agree on are
    the strongest, most defensible candidates to keep.
    """
    merged = ig_ranked.merge(rfe_ranked, on="feature", how="outer")
    ig_top = set(ig_ranked["feature"].head(top_k))
    rfe_top = set(rfe_ranked.loc[rfe_ranked["rfe_rank"] <= top_k, "feature"])
    agreed = sorted(ig_top & rfe_top)
    union = sorted(ig_top | rfe_top)
    return merged, agreed, union


def apply_selection(X_train, X_test, selected_features: list):
    """Trim train/test feature matrices down to the selected feature list."""
    return X_train[selected_features], X_test[selected_features]
