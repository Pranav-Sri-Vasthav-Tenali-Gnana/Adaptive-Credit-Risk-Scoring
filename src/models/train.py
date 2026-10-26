import pandas as pd
import numpy as np
import yaml
import logging
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

import mlflow
import mlflow.xgboost

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_window_data(year, config):
    """Load a specific yearly window."""
    path = Path(config["data"]["windows_path"]) / f"window_{year}.csv"
    df = pd.read_csv(path)
    X = df.drop(columns=["target", "year"], errors="ignore")
    y = df["target"]
    return X, y


def load_multi_window_data(years, config):
    """Load and concatenate multiple yearly windows (for sliding window training)."""
    dfs = []
    for year in years:
        path = Path(config["data"]["windows_path"]) / f"window_{year}.csv"
        if path.exists():
            dfs.append(pd.read_csv(path))
    df = pd.concat(dfs, ignore_index=True)
    X = df.drop(columns=["target", "year"], errors="ignore")
    y = df["target"]
    return X, y


def get_models(config):
    """Instantiate all three models from config."""
    xgb_params = config["models"]["xgboost"].copy()
    xgb_params.pop("random_state", None)

    models = {
        "logistic_regression": LogisticRegression(**config["models"]["logistic_regression"]),
        "lightgbm": LGBMClassifier(**config["models"]["lightgbm"]),
        "xgboost": XGBClassifier(**config["models"]["xgboost"]),
    }
    return models


def train_and_evaluate(X, y, model_name, model, config, year, run_name=None):
    """Train a single model and log results to MLflow."""
    cfg = config["data"]
    test_size = cfg["test_size"]
    random_state = cfg["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    try:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(f"  SMOTE applied - new train size: {len(X_train)}")
    except Exception as e:
        logger.warning(f"  SMOTE failed: {e} - using original data")

    if model_name == "logistic_regression":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    with mlflow.start_run(run_name=run_name or f"{model_name}_year_{year}"):
        mlflow.log_param("model", model_name)
        mlflow.log_param("year", year)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        logger.info(f"  Training {model_name} on year {year}...")
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("f1_score", f1)
        logger.info(f"  {model_name} - AUC: {auc:.4f} | F1: {f1:.4f}")

        output_dir = Path(config["outputs"]["models_path"])
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{model_name}_year_{year}.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))

        # Log model to registry (only XGBoost). The MLflow file store does not
        # support the Model Registry, so fall back to logging the model artifact
        # without registration rather than failing the whole run.
        if model_name == "xgboost":
            try:
                mlflow.xgboost.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=config["mlflow"]["model_registry_name"]
                )
            except Exception as e:
                logger.warning(f"  Model registry unavailable ({e}); logging model without registration")
                mlflow.xgboost.log_model(model, artifact_path="model")

        return {
            "model_name": model_name,
            "year": year,
            "auc": auc,
            "f1": f1,
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
        }


def train_baseline(config_path="configs/config.yaml", year=None):
    """Train all models on the first (baseline) year."""
    config = load_config(config_path)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    baseline_year = year or config["data"]["start_year"]
    X, y = load_window_data(baseline_year, config)
    logger.info(f"Training baseline models on year {baseline_year} - {len(X)} samples")

    models = get_models(config)
    results = {}

    for model_name, model in models.items():
        result = train_and_evaluate(X, y, model_name, model, config, baseline_year)
        results[model_name] = result

    logger.info("\nBaseline training complete!")
    for name, r in results.items():
        logger.info(f"  {name}: AUC={r['auc']:.4f} | F1={r['f1']:.4f}")

    return results


def train_adaptive(years, config_path="configs/config.yaml"):
    """
    Sliding window training - train XGBoost on most recent N years.
    Called automatically when drift is detected.
    """
    config = load_config(config_path)
    window_size = config["drift"]["window_size"]
    train_years = years[-window_size:]  # Use last N years

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    logger.info(f"Adaptive retraining on years: {train_years}")
    X, y = load_multi_window_data(train_years, config)

    model = XGBClassifier(**config["models"]["xgboost"])
    result = train_and_evaluate(
        X, y, "xgboost", model, config,
        year=f"adaptive_{train_years[0]}_{train_years[-1]}",
        run_name=f"adaptive_retrain_{train_years[-1]}"
    )

    logger.info(f"Adaptive retraining complete - AUC: {result['auc']:.4f}")
    return result


if __name__ == "__main__":
    results = train_baseline()
