"""
advanced_detectors.py
---------------------
Drift-detection methods that complement the PSI + KS tests in detector.py, so the
project does not rely on a single statistic family. Both methods below use only
scikit-learn and scipy (no extra dependencies).

1. Jensen-Shannon divergence (per feature)
   A bounded, symmetric distributional distance in [0, 1]. Unlike PSI (which is
   unbounded and asymmetric), JS distance is easy to compare across features and
   saturates, so a single wildly-shifted bin cannot dominate. Used here as a
   second opinion on the per-feature covariate drift that PSI flags.

2. Classifier two-sample test (a.k.a. domain classifier / C2ST)
   Label reference rows 0 and current rows 1 and train a classifier to tell them
   apart with cross-validation. If the distributions are identical the best any
   classifier can do is chance (ROC-AUC = 0.5); the further the cross-validated
   AUC rises above 0.5, the more the *joint* feature distribution has drifted.
   This catches multivariate drift (correlations between features changing) that
   per-feature PSI/KS/JS can miss, and its feature importances rank which columns
   drive the drift.
"""

import logging

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

logger = logging.getLogger(__name__)


def js_distance_for_feature(reference, current, bins=10, epsilon=1e-6):
    """Jensen-Shannon *distance* (in [0, 1], base-2) between two 1-D samples.

    Both samples are binned on the reference distribution's deciles - the same
    binning scheme PSI uses - so JS and PSI are directly comparable per feature.
    """
    reference = np.asarray(reference, dtype=float).flatten()
    current = np.asarray(current, dtype=float).flatten()
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    breakpoints = np.unique(np.percentile(reference, np.linspace(0, 100, bins + 1)))
    if len(breakpoints) < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)
    ref_p = ref_counts / max(ref_counts.sum(), 1) + epsilon
    cur_p = cur_counts / max(cur_counts.sum(), 1) + epsilon
    # scipy returns the JS distance (sqrt of the divergence); base=2 -> [0, 1].
    return float(jensenshannon(ref_p, cur_p, base=2))


def js_distance_all_features(reference_df, current_df, features, bins=10):
    """JS distance for every feature present in both frames. Returns {feature: dist}."""
    out = {}
    for f in features:
        if f in reference_df.columns and f in current_df.columns:
            out[f] = round(js_distance_for_feature(
                reference_df[f].values, current_df[f].values, bins=bins), 4)
    return out


def classifier_drift_test(reference_df, current_df, feature_cols,
                          max_per_class=20000, n_splits=5, random_state=42):
    """Detect multivariate covariate drift via a reference-vs-current classifier.

    Returns a dict with the cross-validated ROC-AUC (0.5 = no drift, 1.0 = fully
    separable), a coarse drift flag, and the features the classifier leaned on
    most (i.e. the columns that drive the drift).
    """
    # Balanced subsample so the AUC is not inflated by class-size imbalance and
    # the test stays fast on large windows.
    n = min(len(reference_df), len(current_df), max_per_class)
    rng = np.random.RandomState(random_state)
    ref = reference_df.sample(n, random_state=rng) if len(reference_df) > n else reference_df
    cur = current_df.sample(n, random_state=rng) if len(current_df) > n else current_df

    cols = [c for c in feature_cols if c in ref.columns and c in cur.columns]
    X = pd.concat([ref[cols], cur[cols]], ignore_index=True).astype(np.float32)
    y = np.r_[np.zeros(len(ref)), np.ones(len(cur))]

    clf = HistGradientBoostingClassifier(
        max_iter=150, max_depth=4, learning_rate=0.1, random_state=random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    drift_auc = float(aucs.mean())

    # HistGradientBoostingClassifier has no built-in feature importances, and
    # adding a separate model just to rank features isn't worth the extra
    # dependency - classifier_drift_top_features below does that ranking
    # instead, with plain per-feature AUC.
    return {
        "method": "classifier_two_sample_test",
        "drift_auc": round(drift_auc, 4),
        "drift_auc_std": round(float(aucs.std()), 4),
        "n_per_class": int(n),
        # AUC >= 0.55 is a common, conservative "distributions are distinguishable"
        # cut-off for a balanced C2ST; 0.5 is the no-drift null.
        "drift_detected": bool(drift_auc >= 0.55),
    }


def classifier_drift_top_features(reference_df, current_df, feature_cols,
                                  max_per_class=20000, top_k=8, random_state=42):
    """Rank the features most responsible for ref-vs-current separation.

    Uses per-feature univariate ROC-AUC of a single split (how well each feature
    alone distinguishes the two periods) - dependency-light and interpretable.
    """
    from sklearn.metrics import roc_auc_score

    n = min(len(reference_df), len(current_df), max_per_class)
    rng = np.random.RandomState(random_state)
    ref = reference_df.sample(n, random_state=rng) if len(reference_df) > n else reference_df
    cur = current_df.sample(n, random_state=rng) if len(current_df) > n else current_df
    cols = [c for c in feature_cols if c in ref.columns and c in cur.columns]
    y = np.r_[np.zeros(len(ref)), np.ones(len(cur))]

    scores = {}
    for c in cols:
        vals = pd.concat([ref[c], cur[c]], ignore_index=True).astype(float).values
        # |AUC - 0.5| measures separability in either direction.
        try:
            auc = roc_auc_score(y, vals)
        except ValueError:
            continue
        scores[c] = round(abs(auc - 0.5) + 0.5, 4)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [{"feature": f, "separation_auc": s} for f, s in ranked]
