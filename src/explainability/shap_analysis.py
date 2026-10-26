import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import yaml
import logging
import joblib
from pathlib import Path
from scipy.stats import kendalltau
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_shap_values(model, X, n_background=100):
    """
    Compute SHAP values for a given model and dataset.
    Uses TreeExplainer (fast, exact for tree-based models).

    Returns:
        shap_values: np.array of shape (n_samples, n_features)
        explainer: shap.TreeExplainer object
    """
    logger.info(f"  Computing SHAP values for {len(X)} samples...")
    background = X.sample(min(n_background, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model, background)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # binary classification - take positive class

    return shap_values, explainer


def mean_abs_shap(shap_values, feature_names):
    """
    Compute mean absolute SHAP value per feature.
    Returns a Series sorted by importance.
    """
    mean_shap = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_shap, index=feature_names).sort_values(ascending=False)


def cosine_similarity_shap(shap_vec_a, shap_vec_b):
    """Cosine similarity between two SHAP importance vectors."""
    a = shap_vec_a.values.reshape(1, -1)
    b = shap_vec_b.values.reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0])


def kendall_tau_shap(shap_vec_a, shap_vec_b):
    """Kendall tau rank correlation between two SHAP importance vectors."""
    # Rank both vectors
    rank_a = shap_vec_a.rank(ascending=False)
    rank_b = shap_vec_b.rank(ascending=False)
    # Align features
    common = shap_vec_a.index.intersection(shap_vec_b.index)
    tau, p_value = kendalltau(rank_a[common], rank_b[common])
    return round(tau, 4), round(p_value, 6)


def jaccard_top_n(shap_vec_a, shap_vec_b, n=10):
    """Jaccard overlap of the top-N most important features between two SHAP vectors.

    Each vector is sorted by importance internally before taking the head, so the
    result is correct regardless of the input ordering. (A previous version relied
    on the callers passing pre-sorted Series; when the callers reindexed both
    vectors onto a shared index first, both heads became identical and the metric
    collapsed to a constant 1.0 - masking real top-N churn such as int_rate
    climbing into the top-10.)
    """
    top_a = set(shap_vec_a.sort_values(ascending=False).head(n).index)
    top_b = set(shap_vec_b.sort_values(ascending=False).head(n).index)
    intersection = len(top_a & top_b)
    union = len(top_a | top_b)
    return round(intersection / union, 4) if union > 0 else 0.0


def plot_shap_bar(shap_series, year, output_dir, top_n=15):
    """Plot horizontal bar chart of top SHAP features."""
    top = shap_series.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#D4AC0D" if v > top.mean() else "#7C4DFF" for v in top.values]
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title(f"Feature Importance - Year {year}", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out_path = Path(output_dir) / f"shap_bar_{year}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved SHAP bar chart -> {out_path}")
    return out_path


def plot_stability_over_time(stability_df, output_dir):
    """Plot cosine similarity and Kendall tau over time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(stability_df["year"], stability_df["cosine_similarity"],
             marker="o", color="#D4AC0D", linewidth=2.5)
    ax1.axhline(y=0.95, color="#FF6B6B", linestyle="--", label="Threshold (0.95)")
    ax1.set_title("SHAP Cosine Similarity Over Time", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.plot(stability_df["year"], stability_df["kendall_tau"],
             marker="s", color="#7C4DFF", linewidth=2.5)
    ax2.axhline(y=0.8, color="#FF6B6B", linestyle="--", label="Threshold (0.8)")
    ax2.set_title("SHAP Kendall Tau Rank Correlation Over Time", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Kendall Tau")
    ax2.set_ylim(-0.1, 1.05)
    ax2.legend()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.suptitle("SHAP Explainability Stability Analysis", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = Path(output_dir) / "shap_stability.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP stability plot -> {out_path}")
    return out_path


def run_shap_analysis(config_path="configs/config.yaml"):
    """
    Run full SHAP analysis across all years.
    Computes stability metrics comparing each year to the baseline.
    """
    config = load_config(config_path)
    start_year = config["data"]["start_year"]
    end_year = config["data"]["end_year"]
    n_background = config["shap"]["background_samples"]
    top_n = config["shap"]["top_n_features"]
    plots_dir = Path(config["outputs"]["plots_path"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    baseline_model_path = Path(config["outputs"]["models_path"]) / f"xgboost_year_{start_year}.joblib"
    if not baseline_model_path.exists():
        raise FileNotFoundError(f"Baseline model not found at {baseline_model_path}. Run training first.")
    baseline_model = joblib.load(baseline_model_path)

    baseline_df = pd.read_csv(
        Path(config["data"]["windows_path"]) / f"window_{start_year}.csv"
    )
    X_baseline = baseline_df.drop(columns=["target", "year"], errors="ignore")

    logger.info(f"\nComputing baseline SHAP for year {start_year}...")
    shap_vals_baseline, _ = compute_shap_values(baseline_model, X_baseline, n_background)
    shap_baseline = mean_abs_shap(shap_vals_baseline, X_baseline.columns.tolist())
    plot_shap_bar(shap_baseline, start_year, plots_dir, top_n)

    stability_records = []

    for year in range(start_year + 1, end_year + 1):
        window_path = Path(config["data"]["windows_path"]) / f"window_{year}.csv"
        model_path = Path(config["outputs"]["models_path"]) / f"xgboost_year_{year}.joblib"

        if not window_path.exists():
            continue

        logger.info(f"\nSHAP analysis - Year {year}...")

        year_df = pd.read_csv(window_path)
        X_year = year_df.drop(columns=["target", "year"], errors="ignore")

        # Score this year's data with the BASELINE model, so any drift in the
        # explanations is due to the data shifting, not the model changing.
        shap_vals_year, _ = compute_shap_values(baseline_model, X_year, n_background)
        shap_year = mean_abs_shap(shap_vals_year, X_year.columns.tolist())

        common_features = shap_baseline.index.intersection(shap_year.index)
        sb = shap_baseline[common_features]
        sy = shap_year[common_features]

        cos_sim = cosine_similarity_shap(sb, sy)
        tau, tau_p = kendall_tau_shap(sb, sy)
        jaccard = jaccard_top_n(sb, sy, n=top_n)
        logger.info(f"  Cosine Similarity : {cos_sim:.4f}")
        logger.info(f"  Kendall Tau       : {tau:.4f} (p={tau_p})")
        logger.info(f"  Jaccard Top-{top_n}   : {jaccard:.4f}")

        plot_shap_bar(shap_year, year, plots_dir, top_n)

        stability_records.append({
            "year": year,
            "cosine_similarity": round(cos_sim, 4),
            "kendall_tau": round(tau, 4),
            "kendall_p_value": round(tau_p, 6),
            "jaccard_top_n": round(jaccard, 4),
        })

    if stability_records:
        stability_df = pd.DataFrame(stability_records)
        plot_stability_over_time(stability_df, plots_dir)

        report_dir = Path(config["outputs"]["reports_path"])
        report_dir.mkdir(parents=True, exist_ok=True)
        stability_df.to_csv(report_dir / "shap_stability_report.csv", index=False)
        logger.info(f"\nSHAP stability report saved")
        return stability_df
    else:
        logger.warning("No stability records computed - check data windows")
        return pd.DataFrame()


if __name__ == "__main__":
    run_shap_analysis()
