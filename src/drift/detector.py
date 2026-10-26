"""
detector.py
-----------
Main drift detection pipeline.
Runs PSI + KS tests across all yearly windows and decides
whether to trigger retraining.
"""

import pandas as pd
import numpy as np
import yaml
import logging
import json
from pathlib import Path

from src.drift.psi import calculate_psi_all_features, psi_report
from src.drift.ks_test import ks_test_labels, ks_test_features, ks_report
from src.drift.advanced_detectors import (
    js_distance_all_features, classifier_drift_test, classifier_drift_top_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_window(year, config):
    path = Path(config["data"]["windows_path"]) / f"window_{year}.csv"
    return pd.read_csv(path)


def run_drift_detection(config_path="configs/config.yaml"):
    """
    Run full drift detection across all years.
    Compares each year against the reference (first year).

    Returns:
        drift_log: list of dicts with drift results per year
        retrain_years: list of years where retraining should trigger
    """
    config = load_config(config_path)
    start_year = config["data"]["start_year"]
    end_year = config["data"]["end_year"]
    psi_threshold = config["drift"]["psi_threshold_critical"]
    ks_alpha = config["drift"]["ks_alpha"]

    ref_df = load_window(start_year, config)
    ref_labels = ref_df["target"]
    numerical_features = [c for c in config["features"]["numerical"] if c in ref_df.columns]
    # All model features (numerical + one-hot categoricals) for the multivariate
    # classifier two-sample test; exclude the label and the bookkeeping column.
    model_features = [c for c in ref_df.columns if c not in ("target", "year")]

    drift_log = []
    retrain_years = []

    logger.info(f"\n{'='*60}")
    logger.info(f"Running drift detection: {start_year} -> {end_year}")
    logger.info(f"Reference year: {start_year}")
    logger.info(f"{'='*60}")

    for year in range(start_year + 1, end_year + 1):
        try:
            cur_df = load_window(year, config)
            cur_labels = cur_df["target"]

            logger.info(f"\n--- Year {year} vs {start_year} ---")

            psi_dict = calculate_psi_all_features(ref_df, cur_df, numerical_features)
            flagged_features = psi_report(psi_dict, threshold_critical=psi_threshold)
            mean_psi = np.mean(list(psi_dict.values()))

            ks_result = ks_test_labels(ref_labels, cur_labels, alpha=ks_alpha)
            ks_report(ks_result)

            # Two more independent checks, so the trigger doesn't rest on one
            # statistic family: JS gives a bounded second opinion on PSI, and the
            # classifier test catches multivariate drift PSI/JS can't see.
            js_dict = js_distance_all_features(ref_df, cur_df, numerical_features)
            mean_js = float(np.mean(list(js_dict.values()))) if js_dict else 0.0
            logger.info("  Running classifier two-sample test (domain classifier)...")
            c2st = classifier_drift_test(ref_df, cur_df, model_features)
            c2st_top = classifier_drift_top_features(ref_df, cur_df, model_features)
            logger.info(f"  C2ST drift AUC = {c2st['drift_auc']:.4f} "
                        f"(0.5 = no drift) -> drift={c2st['drift_detected']}")

            # Decision - retrain if ANY detector fires (covariate PSI, label KS, or
            # the multivariate classifier test). This makes the trigger robust to
            # the type of drift, not just label shift.
            psi_drift = len(flagged_features) > 0
            drift_triggered = psi_drift or ks_result["drift_detected"] or c2st["drift_detected"]

            entry = {
                "year": year,
                "mean_psi": round(float(mean_psi), 4),
                "n_flagged_features": len(flagged_features),
                "flagged_features": flagged_features,
                "psi_per_feature": {k: float(v) for k, v in psi_dict.items()},
                "ks_statistic": float(ks_result["ks_statistic"]),
                "ks_p_value": float(ks_result["p_value"]),
                "label_drift": bool(ks_result["drift_detected"]),
                "mean_js_distance": round(mean_js, 4),
                "js_per_feature": js_dict,
                "classifier_drift_auc": c2st["drift_auc"],
                "classifier_drift_auc_std": c2st["drift_auc_std"],
                "classifier_drift_detected": c2st["drift_detected"],
                "classifier_top_features": c2st_top,
                # Which individual detectors fired - lets us compare trigger policies.
                "psi_drift": bool(psi_drift),
                "retrain_triggered": bool(drift_triggered),
                "reference_default_rate": float(ks_result["reference_default_rate"]),
                "current_default_rate": float(ks_result["current_default_rate"]),
            }
            drift_log.append(entry)

            if drift_triggered:
                retrain_years.append(year)
                logger.info(f"  DRIFT TRIGGERED for year {year} - retraining recommended")
            else:
                logger.info(f"  No drift triggered for year {year}")

        except FileNotFoundError:
            logger.warning(f"  Window file for year {year} not found - skipping")

    # Save drift log
    report_dir = Path(config["outputs"]["reports_path"])
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "drift_report.json"
    with open(report_path, "w") as f:
        json.dump(drift_log, f, indent=2)
    logger.info(f"\nDrift report saved to {report_path}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("DRIFT DETECTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Year':<6} {'MeanPSI':>8} {'MeanJS':>8} {'C2ST-AUC':>9} "
                f"{'Flagged':>8} {'LabelDrift':>11} {'Retrain':>8}")
    logger.info("-" * 70)
    for entry in drift_log:
        logger.info(
            f"{entry['year']:<6} {entry['mean_psi']:>8.4f} "
            f"{entry['mean_js_distance']:>8.4f} {entry['classifier_drift_auc']:>9.4f} "
            f"{entry['n_flagged_features']:>8} "
            f"{'YES' if entry['label_drift'] else 'NO':>11} "
            f"{'YES' if entry['retrain_triggered'] else 'NO':>8}"
        )

    logger.info(f"\nYears requiring retraining: {retrain_years}")
    return drift_log, retrain_years


if __name__ == "__main__":
    drift_log, retrain_years = run_drift_detection()
