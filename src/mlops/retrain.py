"""
retrain.py
----------
Automated retraining pipeline: for each year flagged by detector.py, retrain
XGBoost on a sliding window of recent years and compare it against the static
baseline. This is the earlier, simpler version of the comparison - it scores
the adaptive model on a mixed test split rather than a leakage-free held-out
one, which is why the run_phase4/5 drivers use their own evaluation instead.
Kept here for the legacy main.py path.
"""

import pandas as pd
import numpy as np
import yaml
import logging
import joblib
import json
from pathlib import Path

import mlflow

from src.models.train import train_adaptive, load_window_data
from src.models.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_all_available_years(config):
    """Return sorted list of years that have window files."""
    windows_path = Path(config["data"]["windows_path"])
    years = []
    for f in windows_path.glob("window_*.csv"):
        try:
            year = int(f.stem.split("_")[1])
            years.append(year)
        except Exception:
            pass
    return sorted(years)


def load_static_model(year, config):
    """Load a previously trained static model for a given year."""
    model_path = Path(config["outputs"]["models_path"]) / f"xgboost_year_{year}.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None


def adaptive_retraining_pipeline(drift_log, config_path="configs/config.yaml"):
    """For each year flagged in drift_log, train on the sliding window and
    compare against the static baseline. Returns a list of comparison dicts."""
    config = load_config(config_path)
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    all_years = get_all_available_years(config)
    retraining_results = []

    for entry in drift_log:
        if not entry["retrain_triggered"]:
            continue

        year = entry["year"]
        logger.info(f"\n{'='*60}")
        logger.info(f"Adaptive retraining triggered for year {year}")
        logger.info(f"{'='*60}")

        years_up_to_now = [y for y in all_years if y <= year]
        adaptive_result = train_adaptive(years_up_to_now, config_path)
        X_test, y_test = load_window_data(year, config)

        static_model = load_static_model(config["data"]["start_year"], config)
        if static_model is not None:
            static_metrics = evaluate_model(static_model, X_test, y_test, "static")
        else:
            static_metrics = {"auc": None, "f1": None}

        adaptive_metrics = {
            "auc": adaptive_result["auc"],
            "f1": adaptive_result["f1"]
        }

        improvement = None
        if static_metrics["auc"] and adaptive_metrics["auc"]:
            improvement = adaptive_metrics["auc"] - static_metrics["auc"]

        result = {
            "year": year,
            "static_auc": static_metrics["auc"],
            "adaptive_auc": adaptive_metrics["auc"],
            "static_f1": static_metrics["f1"],
            "adaptive_f1": adaptive_metrics["f1"],
            "auc_improvement": round(improvement, 4) if improvement else None,
            "retrain_years_used": years_up_to_now,
        }
        retraining_results.append(result)

        logger.info(f"\n  Results for Year {year}:")
        logger.info(f"  Static model AUC    : {static_metrics['auc']}")
        logger.info(f"  Adaptive model AUC  : {adaptive_metrics['auc']:.4f}")
        if improvement:
            logger.info(f"  AUC Improvement     : +{improvement:.4f}")

    report_dir = Path(config["outputs"]["reports_path"])
    report_dir.mkdir(parents=True, exist_ok=True)
    results_path = report_dir / "retraining_results.json"
    with open(results_path, "w") as f:
        json.dump(retraining_results, f, indent=2)

    logger.info(f"\nRetraining pipeline complete - results saved to {results_path}")
    return retraining_results


if __name__ == "__main__":
    # Load drift log from previous run
    try:
        with open("outputs/reports/drift_report.json") as f:
            drift_log = json.load(f)
        adaptive_retraining_pipeline(drift_log)
    except FileNotFoundError:
        logger.error("Drift report not found. Run detector.py first.")
