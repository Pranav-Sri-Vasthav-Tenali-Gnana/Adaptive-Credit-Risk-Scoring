"""
ks_test.py
----------
Kolmogorov-Smirnov (KS) Test for label/target drift detection.
Checks whether the distribution of the target variable (default rate)
has significantly changed between two time windows.

If p-value < alpha (0.05) -> distributions are different -> drift detected
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def ks_test_labels(reference_labels, current_labels, alpha=0.05):
    """
    Run KS test between two label distributions.

    Args:
        reference_labels: array-like, labels from reference period
        current_labels:   array-like, labels from current period
        alpha:            significance level (default 0.05)

    Returns:
        dict with ks_statistic, p_value, drift_detected, interpretation
    """
    reference_labels = np.array(reference_labels)
    current_labels = np.array(current_labels)

    ks_stat, p_value = stats.ks_2samp(reference_labels, current_labels)
    drift_detected = p_value < alpha

    result = {
        "ks_statistic": round(ks_stat, 4),
        "p_value": round(p_value, 6),
        "drift_detected": drift_detected,
        "alpha": alpha,
        "reference_default_rate": round(reference_labels.mean(), 4),
        "current_default_rate": round(current_labels.mean(), 4),
    }

    return result


def ks_test_features(reference_df, current_df, features, alpha=0.05):
    """
    Run KS test for each feature between two DataFrames.

    Returns:
        dict: {feature_name: {ks_statistic, p_value, drift_detected}}
    """
    results = {}
    for feature in features:
        if feature in reference_df.columns and feature in current_df.columns:
            ref_vals = reference_df[feature].dropna().values
            cur_vals = current_df[feature].dropna().values
            ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
            results[feature] = {
                "ks_statistic": round(ks_stat, 4),
                "p_value": round(p_value, 6),
                "drift_detected": p_value < alpha
            }
    return results


def ks_report(label_result, feature_results=None):
    """Print a clean KS test report."""
    logger.info("\n=== KS Test Report (Label Drift) ===")
    logger.info(f"  Reference default rate : {label_result['reference_default_rate']:.2%}")
    logger.info(f"  Current default rate   : {label_result['current_default_rate']:.2%}")
    logger.info(f"  KS Statistic           : {label_result['ks_statistic']:.4f}")
    logger.info(f"  P-value                : {label_result['p_value']:.6f}")
    logger.info(f"  Alpha                  : {label_result['alpha']}")

    if label_result["drift_detected"]:
        logger.info("  Label drift DETECTED - default rate has shifted significantly")
    else:
        logger.info("  No significant label drift detected")

    if feature_results:
        logger.info("\n=== KS Test Report (Feature Drift) ===")
        logger.info(f"{'Feature':<30} {'KS Stat':>10} {'P-value':>12} {'Drift?':>8}")
        logger.info("-" * 65)
        for feature, res in sorted(feature_results.items(), key=lambda x: x[1]["ks_statistic"], reverse=True):
            drift_flag = "YES" if res["drift_detected"] else "No"
            logger.info(f"{feature:<30} {res['ks_statistic']:>10.4f} {res['p_value']:>12.6f} {drift_flag:>8}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)

    # Simulate: reference year has 15% default rate, current year has 25%
    ref_labels = np.random.binomial(1, 0.15, 1000)
    cur_labels = np.random.binomial(1, 0.25, 1000)

    result = ks_test_labels(ref_labels, cur_labels)
    ks_report(result)
