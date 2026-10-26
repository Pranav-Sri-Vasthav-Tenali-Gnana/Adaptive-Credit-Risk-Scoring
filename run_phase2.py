import warnings
warnings.filterwarnings("ignore")

import logging
import yaml
from pathlib import Path

import mlflow
from xgboost import XGBClassifier

from src.models.train import train_baseline, load_window_data, train_and_evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "configs/config.yaml"


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    for d in ["outputs/models", "outputs/reports", "outputs/plots", "mlruns"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    start_year = config["data"]["start_year"]
    end_year = config["data"]["end_year"]

    logger.info("=" * 60)
    logger.info(f"BASELINE TRAINING - all 3 models on year {start_year}")
    logger.info("=" * 60)
    baseline_results = train_baseline(CONFIG_PATH, year=start_year)

    # One XGBoost per later year too, so Phase 3 can measure how each year's
    # own model would have performed (the adaptive-comparison baseline).
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    per_year_results = {}
    for year in range(start_year + 1, end_year + 1):
        window = Path(config["data"]["windows_path"]) / f"window_{year}.csv"
        if not window.exists():
            logger.warning(f"  Skipping {year}: {window} not found")
            continue
        logger.info("=" * 60)
        logger.info(f"PER-YEAR XGBOOST - year {year}")
        logger.info("=" * 60)
        X, y = load_window_data(year, config)
        model = XGBClassifier(**config["models"]["xgboost"])
        per_year_results[year] = train_and_evaluate(X, y, "xgboost", model, config, year)

    print("\n" + "=" * 64)
    print(f"PHASE 2 SUMMARY - baseline year {start_year}")
    print("=" * 64)
    print(f"{'Model':<22}{'AUC-ROC':>12}{'F1':>12}")
    print("-" * 64)
    for name, r in baseline_results.items():
        print(f"{name:<22}{r['auc']:>12.4f}{r['f1']:>12.4f}")

    best = max(baseline_results.items(), key=lambda kv: kv[1]["auc"])
    print("-" * 64)
    print(f"BEST baseline model (by AUC-ROC): {best[0]} "
          f"(AUC={best[1]['auc']:.4f}, F1={best[1]['f1']:.4f})")

    if per_year_results:
        print("\nPer-year XGBoost models (for adaptive comparison):")
        print(f"{'Year':<10}{'AUC-ROC':>12}{'F1':>12}")
        print("-" * 34)
        # include the baseline-year xgboost too, for a full per-year view
        print(f"{start_year:<10}{baseline_results['xgboost']['auc']:>12.4f}"
              f"{baseline_results['xgboost']['f1']:>12.4f}")
        for year, r in per_year_results.items():
            print(f"{year:<10}{r['auc']:>12.4f}{r['f1']:>12.4f}")

    print("\nModels saved to outputs/models/  |  MLflow logs in mlruns/")
    print("=" * 64)


if __name__ == "__main__":
    main()
