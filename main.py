"""
main.py
-------
Full end-to-end pipeline runner for:
"Adaptive Credit Risk Scoring with Concept Drift Detection"

Author : Pranav Sri Vasthav Tenali Gnana
ID     : M25DE1069
IIT Jodhpur | M.Tech Data Engineering

Run:
    python main.py
"""

import logging
import json
import yaml
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("outputs/pipeline.log", mode="w")
    ]
)
logger = logging.getLogger(__name__)

BANNER = """
==================================================================
   Adaptive Credit Risk Scoring with Concept Drift Detection
   Author : Pranav Sri Vasthav Tenali Gnana | M25DE1069
   IIT Jodhpur | M.Tech Data Engineering | 2025-2026
==================================================================
"""


def ensure_dirs():
    """Create all necessary output directories."""
    dirs = [
        "data/raw", "data/processed", "data/windows",
        "outputs/models", "outputs/reports", "outputs/plots",
        "mlruns"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def run_pipeline(config_path="configs/config.yaml"):
    print(BANNER)
    ensure_dirs()

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("\n" + "="*60)
    logger.info("PHASE 1: DATA PREPROCESSING")
    logger.info("="*60)

    try:
        from src.data.preprocess import run_preprocessing
        df, years = run_preprocessing(config_path)
        logger.info(f"Phase 1 complete - {len(years)} yearly windows created")
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.error("Please download the Lending Club dataset and place it at:")
        logger.error(f"  {config['data']['raw_path']}")
        logger.error("Download: https://www.kaggle.com/datasets/wordsforthewise/lending-club")
        return

    logger.info("\n" + "="*60)
    logger.info("PHASE 2: BASELINE MODEL TRAINING")
    logger.info("="*60)

    from src.models.train import train_baseline
    baseline_results = train_baseline(config_path)
    logger.info(f"Phase 2 complete - baseline models trained")

    # Also train a model per year for adaptive comparison
    logger.info("\nTraining per-year models for adaptive comparison...")
    from src.models.train import load_window_data, get_models, train_and_evaluate
    import mlflow

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    start_year = config["data"]["start_year"]
    for year in years[1:]:  # Skip first year (already trained as baseline)
        try:
            X, y = load_window_data(year, config)
            from xgboost import XGBClassifier
            model = XGBClassifier(**config["models"]["xgboost"])
            train_and_evaluate(X, y, "xgboost", model, config, year)
        except Exception as e:
            logger.warning(f"  Could not train model for year {year}: {e}")

    logger.info("\n" + "="*60)
    logger.info("PHASE 3: DRIFT DETECTION")
    logger.info("="*60)

    from src.drift.detector import run_drift_detection
    drift_log, retrain_years = run_drift_detection(config_path)
    logger.info(f"Phase 3 complete - drift detected in years: {retrain_years}")

    logger.info("\n" + "="*60)
    logger.info("PHASE 4: ADAPTIVE RETRAINING + SHAP ANALYSIS")
    logger.info("="*60)

    from src.mlops.retrain import adaptive_retraining_pipeline
    retraining_results = adaptive_retraining_pipeline(drift_log, config_path)
    logger.info(f"Adaptive retraining complete")

    try:
        from src.explainability.shap_analysis import run_shap_analysis
        stability_df = run_shap_analysis(config_path)
        logger.info(f"SHAP analysis complete")
    except Exception as e:
        logger.warning(f"SHAP analysis failed: {e}")

    logger.info("\n" + "="*60)
    logger.info("PHASE 5: BACKTESTING")
    logger.info("="*60)

    try:
        from src.models.evaluate import backtest_over_years
        backtest_df = backtest_over_years(config_path)
        logger.info(f"Backtesting complete")
    except Exception as e:
        logger.warning(f"Backtesting failed: {e}")

    logger.info("\n" + "="*60)
    logger.info("FULL PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info("\nOutputs:")
    logger.info("  Drift report     -> outputs/reports/drift_report.json")
    logger.info("  SHAP report      -> outputs/reports/shap_stability_report.csv")
    logger.info("  Backtest results -> outputs/reports/backtest_results.csv")
    logger.info("  Plots            -> outputs/plots/")
    logger.info("  Models           -> outputs/models/")
    logger.info("  MLflow UI        -> run: mlflow ui  (then open localhost:5000)")


if __name__ == "__main__":
    run_pipeline()
