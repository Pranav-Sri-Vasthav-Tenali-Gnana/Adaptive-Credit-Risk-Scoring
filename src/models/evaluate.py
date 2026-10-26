import json
import pandas as pd
import numpy as np
import yaml
import logging
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, roc_curve, average_precision_score,
)
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_model(model, X, y, model_name="model", threshold=0.5):
    """Evaluate a model on given data. Returns dict of metrics.

    Adds PR-AUC (average precision) - the threshold-free metric appropriate for an
    imbalanced positive class - and reports F1 at both 0.5 and a supplied tuned
    threshold so the rising-base-rate artifact at 0.5 is visible, not hidden.
    """
    try:
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        y_pred_tuned = (y_pred_proba >= threshold).astype(int)

        return {
            "model": model_name,
            "auc": round(roc_auc_score(y, y_pred_proba), 4),
            "pr_auc": round(average_precision_score(y, y_pred_proba), 4),
            "f1": round(f1_score(y, y_pred), 4),
            "f1_tuned": round(f1_score(y, y_pred_tuned), 4),
            "precision": round(precision_score(y, y_pred), 4),
            "recall": round(recall_score(y, y_pred), 4),
            "threshold": round(float(threshold), 4),
        }
    except Exception as e:
        logger.error(f"Evaluation failed for {model_name}: {e}")
        return {"model": model_name, "auc": None, "pr_auc": None, "f1": None}


def backtest_over_years(config_path="configs/config.yaml"):
    """
    Backtest static vs adaptive models across all yearly windows.

    This is the key result chart for the paper -
    shows static model degrading and adaptive model staying stable.
    """
    config = load_config(config_path)
    windows_path = Path(config["data"]["windows_path"])
    models_path = Path(config["outputs"]["models_path"])
    plots_dir = Path(config["outputs"]["plots_path"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    start_year = config["data"]["start_year"]
    end_year = config["data"]["end_year"]
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    # Load static baseline model (trained only on first year)
    static_model_path = models_path / f"xgboost_year_{start_year}.joblib"
    if not static_model_path.exists():
        raise FileNotFoundError(f"Static model not found at {static_model_path}")
    static_model = joblib.load(static_model_path)

    # Tuned decision thresholds from Phase 4 (validation-tuned, no test leakage).
    thr_path = models_path / "backtest_thresholds.json"
    thresholds = json.loads(thr_path.read_text()) if thr_path.exists() else {}
    static_thr = thresholds.get("static", 0.5)

    results = []

    for year in range(start_year, end_year + 1):
        window_path = windows_path / f"window_{year}.csv"
        if not window_path.exists():
            continue

        df = pd.read_csv(window_path)
        X = df.drop(columns=["target", "year"], errors="ignore")
        y = df["target"]

        # Evaluate on the same held-out 20% test split (random_state-matched)
        # that Phase 4 used, so the sliding-window adaptive models are scored
        # leakage-free and the plot matches the headline comparison table.
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Static-2015 model on this year's held-out test set
        static_metrics = evaluate_model(static_model, X_test, y_test, "static", static_thr)

        # Sliding-window adaptive model for this year (Phase 4). The baseline
        # year has no adaptive model - the static model IS the model at t0.
        adaptive_model_path = models_path / f"xgboost_adaptive_{year}.joblib"
        if adaptive_model_path.exists():
            adaptive_model = joblib.load(adaptive_model_path)
            adaptive_thr = thresholds.get(str(year), 0.5)
            adaptive_metrics = evaluate_model(
                adaptive_model, X_test, y_test, "adaptive", adaptive_thr)
        else:
            adaptive_metrics = static_metrics.copy()
            adaptive_metrics["model"] = "adaptive"

        results.append({
            "year": year,
            "static_auc": static_metrics["auc"],
            "adaptive_auc": adaptive_metrics["auc"],
            "static_pr_auc": static_metrics["pr_auc"],
            "adaptive_pr_auc": adaptive_metrics["pr_auc"],
            "static_f1": static_metrics["f1"],
            "adaptive_f1": adaptive_metrics["f1"],
            "static_f1_tuned": static_metrics["f1_tuned"],
            "adaptive_f1_tuned": adaptive_metrics["f1_tuned"],
        })

        logger.info(
            f"Year {year} - Static AUC: {static_metrics['auc']} | "
            f"Adaptive AUC: {adaptive_metrics['auc']}"
        )

    results_df = pd.DataFrame(results)

    # Plot AUC over time
    _plot_auc_over_time(results_df, plots_dir)
    _plot_prauc_over_time(results_df, plots_dir)
    _plot_f1_over_time(results_df, plots_dir)

    # Save
    report_dir = Path(config["outputs"]["reports_path"])
    report_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(report_dir / "backtest_results.csv", index=False)
    logger.info(f"\nBacktest results saved")

    return results_df


def _plot_auc_over_time(df, output_dir):
    """Plot static vs adaptive AUC over years - the key paper figure."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df["year"], df["static_auc"] * 100,
            marker="o", color="#FF6B6B", linewidth=2.5, label="Static Model (trained once)")
    ax.plot(df["year"], df["adaptive_auc"] * 100,
            marker="s", color="#00BCD4", linewidth=2.5, label="Adaptive Model (drift-triggered retraining)")

    ax.fill_between(df["year"],
                    df["static_auc"] * 100,
                    df["adaptive_auc"] * 100,
                    alpha=0.15, color="#7C4DFF", label="Performance Gap")

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("AUC-ROC (%)", fontsize=12)
    ax.set_title("Static vs Adaptive Model: AUC-ROC Over Time\n(Key Research Finding)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(50, 100)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = Path(output_dir) / "auc_over_time.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved AUC plot -> {out_path}")


def _plot_prauc_over_time(df, output_dir):
    """Plot static vs adaptive PR-AUC (average precision) over years - the proper
    threshold-free metric for the imbalanced positive (default) class."""
    if "static_pr_auc" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["year"], df["static_pr_auc"], marker="o", color="#FF6B6B",
            linewidth=2.5, label="Static Model (trained once)")
    ax.plot(df["year"], df["adaptive_pr_auc"], marker="s", color="#00BCD4",
            linewidth=2.5, label="Adaptive Model (drift-triggered retraining)")
    ax.fill_between(df["year"], df["static_pr_auc"], df["adaptive_pr_auc"],
                    alpha=0.15, color="#7C4DFF", label="PR-AUC Gap")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("PR-AUC (Average Precision)", fontsize=12)
    ax.set_title("Static vs Adaptive Model: PR-AUC Over Time\n"
                 "(threshold-free metric for the imbalanced default class)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out_path = Path(output_dir) / "pr_auc_over_time.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved PR-AUC plot -> {out_path}")


def _plot_f1_over_time(df, output_dir):
    """Plot static vs adaptive F1 over years - at the tuned threshold (solid) and
    at the naive 0.5 cutoff (dashed) so the base-rate artifact is explicit."""
    fig, ax = plt.subplots(figsize=(12, 6))

    has_tuned = "static_f1_tuned" in df.columns
    if has_tuned:
        ax.plot(df["year"], df["static_f1_tuned"], marker="o", color="#FF6B6B",
                linewidth=2.5, label="Static - F1 @ tuned threshold")
        ax.plot(df["year"], df["adaptive_f1_tuned"], marker="s", color="#00BCD4",
                linewidth=2.5, label="Adaptive - F1 @ tuned threshold")
    ax.plot(df["year"], df["static_f1"], marker="o", color="#FF6B6B", alpha=0.5,
            linestyle="--", linewidth=1.8, label="Static - F1 @ 0.5 (artifact)")
    ax.plot(df["year"], df["adaptive_f1"], marker="s", color="#00BCD4", alpha=0.5,
            linestyle="--", linewidth=1.8, label="Adaptive - F1 @ 0.5 (artifact)")

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Static vs Adaptive Model: F1 Score Over Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = Path(output_dir) / "f1_over_time.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved F1 plot -> {out_path}")


if __name__ == "__main__":
    backtest_over_years()
