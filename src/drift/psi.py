"""
psi.py
------
Population Stability Index (PSI) calculation.
Measures how much a feature's distribution has shifted
between a reference period (training) and current period.

PSI < 0.10  -> No significant change
PSI 0.10-0.20 -> Moderate change (warning)
PSI > 0.20  -> Significant change (retrain!)
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def calculate_psi_for_feature(reference, current, bins=10, epsilon=1e-6):
    """
    Calculate PSI between reference and current distributions for one feature.

    Args:
        reference: array-like, reference distribution (e.g. 2015 data)
        current:   array-like, current distribution (e.g. 2020 data)
        bins:      number of bins for bucketing
        epsilon:   small value to avoid log(0)

    Returns:
        psi_value: float
    """
    reference = np.array(reference).flatten()
    current = np.array(current).flatten()
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Bin edges come from the reference distribution's deciles, so both periods
    # get bucketed on the same scale.
    breakpoints = np.unique(np.percentile(reference, np.linspace(0, 100, bins + 1)))
    if len(breakpoints) < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)
    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    # epsilon keeps empty bins from producing log(0)
    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    cur_pct = np.where(cur_pct == 0, epsilon, cur_pct)

    return np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))


def calculate_psi_all_features(reference_df, current_df, features, bins=10):
    """
    Calculate PSI for all features between two DataFrames.

    Returns:
        dict: {feature_name: psi_value}
    """
    psi_results = {}
    for feature in features:
        if feature in reference_df.columns and feature in current_df.columns:
            psi = calculate_psi_for_feature(
                reference_df[feature].values,
                current_df[feature].values,
                bins=bins
            )
            psi_results[feature] = round(psi, 4)

    return psi_results


def interpret_psi(psi_value):
    """Return human-readable interpretation of PSI value."""
    if psi_value < 0.1:
        return "Stable - no action needed"
    elif psi_value < 0.2:
        return "Warning - moderate drift detected"
    else:
        return "Critical - significant drift, retrain recommended"


def psi_report(psi_dict, threshold_warning=0.1, threshold_critical=0.2):
    """
    Print a clean PSI report and return flagged features.

    Returns:
        flagged: list of features with PSI > threshold_critical
    """
    flagged = []
    logger.info("\n=== PSI Report ===")
    logger.info(f"{'Feature':<30} {'PSI':>8}  {'Status'}")
    logger.info("-" * 65)

    for feature, psi_val in sorted(psi_dict.items(), key=lambda x: -x[1]):
        status = interpret_psi(psi_val)
        logger.info(f"{feature:<30} {psi_val:>8.4f}  {status}")
        if psi_val >= threshold_critical:
            flagged.append(feature)

    logger.info(f"\nFeatures with critical drift: {len(flagged)}")
    if flagged:
        logger.info(f"  -> {flagged}")

    return flagged


if __name__ == "__main__":
    # Quick test with synthetic data
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)

    ref = pd.DataFrame({"loan_amnt": np.random.normal(10000, 2000, 1000),
                         "dti": np.random.normal(15, 5, 1000)})
    cur = pd.DataFrame({"loan_amnt": np.random.normal(12000, 3000, 1000),  # shifted
                         "dti": np.random.normal(15.2, 5, 1000)})           # stable

    psi_dict = calculate_psi_all_features(ref, cur, ["loan_amnt", "dti"])
    flagged = psi_report(psi_dict)
    print(f"\nFlagged features: {flagged}")
