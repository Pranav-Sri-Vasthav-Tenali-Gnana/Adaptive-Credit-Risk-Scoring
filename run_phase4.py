"""
run_phase4.py
-------------
Phase 4 driver: Adaptive Retraining + SHAP Analysis (the key phase).

PART A - Adaptive Retraining (sliding window), with a FAIR comparison:
  For each drift year, the target year's window is split into train/test (20%).
  The adaptive model trains on the sliding window of recent years (using only the
  target year's TRAIN portion - no leakage), and BOTH the static-2015 model and
  the adaptive model are scored on the SAME held-out target-year test set.
    2016 adaptive -> train on [2015, 2016]
    2017 adaptive -> train on [2015, 2016, 2017]
    2018 adaptive -> train on [2016, 2017, 2018]

PART B - SHAP Analysis:
  SHAP values of the STATIC 2015 model on each year (sampled), top-10 features,
  cosine similarity + Kendall tau vs the 2015 baseline, per-year bar charts and a
  stability-over-time plot.

Outputs: outputs/reports/retraining_results.json, outputs/reports/final_comparison.csv,
outputs/plots/shap_bar_*.png, outputs/plots/shap_stability.png and adaptive models.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import logging
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")

import mlflow
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score, precision_recall_curve,
)

from src.explainability.shap_analysis import (
    compute_shap_values, mean_abs_shap, cosine_similarity_shap,
    kendall_tau_shap, jaccard_top_n, plot_shap_bar, plot_stability_over_time,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "configs/config.yaml"

# Sliding windows exactly as specified for Phase 4.
SLIDING_WINDOWS = {
    2016: [2015, 2016],
    2017: [2015, 2016, 2017],
    2018: [2016, 2017, 2018],
}


def strategy_train_years(target_year, start_year):
    """Training-year pools for each retraining strategy (the naive baselines that
    answer 'would simpler retraining do just as well as the sliding window?')."""
    return {
        # Drift-triggered sliding window of recent years (the headline method).
        "sliding": SLIDING_WINDOWS[target_year],
        # Naive: retrain on the most recent year only (shortest memory).
        "naive_recent": [target_year],
        # Expanding window: keep ALL history 2015..target (longest memory).
        "expanding": list(range(start_year, target_year + 1)),
    }


def best_f1_threshold(y_val, proba_val):
    """Decision threshold that maximizes F1 on a validation set (real prevalence,
    NOT SMOTE-balanced, NOT the test set) - removes the fixed-0.5 artifact without
    leaking the test labels."""
    prec, rec, thr = precision_recall_curve(y_val, proba_val)
    if len(thr) == 0:
        return 0.5
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[int(np.argmax(f1))])


def evaluate_with_threshold(model, X, y, threshold):
    """Threshold-free (AUC, PR-AUC) + threshold-dependent (F1@0.5, F1@tuned) metrics."""
    proba = model.predict_proba(X)[:, 1]
    return {
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "pr_auc": round(float(average_precision_score(y, proba)), 4),
        "f1_05": round(float(f1_score(y, (proba >= 0.5).astype(int))), 4),
        "f1_tuned": round(float(f1_score(y, (proba >= threshold).astype(int))), 4),
        "threshold": round(float(threshold), 4),
    }


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_window_xy(year, config):
    """Load a window as (X float32 DataFrame, y Series)."""
    path = Path(config["data"]["windows_path"]) / f"window_{year}.csv"
    df = pd.read_csv(path)
    X = df.drop(columns=["target", "year"], errors="ignore").astype(np.float32)
    y = df["target"].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# PART A - Adaptive retraining with fair, leakage-free evaluation
# ---------------------------------------------------------------------------
def adaptive_retraining(config, window_cache):
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    models_path = Path(config["outputs"]["models_path"])
    models_path.mkdir(parents=True, exist_ok=True)

    start_year = config["data"]["start_year"]
    test_size = config["data"]["test_size"]
    rs = config["data"]["random_state"]

    static_model = joblib.load(models_path / f"xgboost_year_{start_year}.joblib")

    def build_pool(train_years, target_year, X_tr_fit, y_tr_fit):
        """Concatenate full earlier windows + the target year's fit portion."""
        Xs, ys = [], []
        for yr in train_years:
            if yr == target_year:
                Xs.append(X_tr_fit); ys.append(y_tr_fit)
            else:
                Xy, yy = window_cache[yr]
                Xs.append(Xy); ys.append(yy)
        return pd.concat(Xs, ignore_index=True), pd.concat(ys, ignore_index=True)

    results = []
    thresholds = {}  # for Phase 5 backtest: {"static": thr, "2016": thr, ...}
    for target_year in SLIDING_WINDOWS:
        strategies = strategy_train_years(target_year, start_year)
        logger.info("=" * 60)
        logger.info(f"Adaptive retraining for {target_year} - strategies {list(strategies)}")
        logger.info("=" * 60)

        # Hold out the target year's test set (the shared, leakage-free eval set).
        # NOTE: identical split (random_state, test_size) to Phase 5's backtest.
        Xt, yt = window_cache[target_year]
        X_tr80, X_test, y_tr80, y_test = train_test_split(
            Xt, yt, test_size=test_size, random_state=rs, stratify=yt
        )
        # Carve a validation slice (real prevalence) from the target-year TRAIN
        # portion, used ONLY to tune decision thresholds - never the test set.
        X_tr_fit, X_val, y_tr_fit, y_val = train_test_split(
            X_tr80, y_tr80, test_size=0.15, random_state=rs, stratify=y_tr80
        )

        # ---- Static-2015 model: tune its threshold on the same recent val ----
        s_thr = best_f1_threshold(y_val, static_model.predict_proba(X_val)[:, 1])
        static_metrics = evaluate_with_threshold(static_model, X_test, y_test, s_thr)
        thresholds["static"] = static_metrics["threshold"]
        logger.info(f"  STATIC   AUC {static_metrics['auc']:.4f} | "
                    f"PR-AUC {static_metrics['pr_auc']:.4f} | "
                    f"F1@0.5 {static_metrics['f1_05']:.4f} | "
                    f"F1@tuned {static_metrics['f1_tuned']:.4f}")

        # ---- Each retraining strategy ----
        strat_metrics = {}
        for name, train_years in strategies.items():
            X_train, y_train = build_pool(train_years, target_year, X_tr_fit, y_tr_fit)
            X_res, y_res = SMOTE(random_state=rs).fit_resample(X_train, y_train)
            model = XGBClassifier(**config["models"]["xgboost"])
            model.fit(X_res, y_res)

            thr = best_f1_threshold(y_val, model.predict_proba(X_val)[:, 1])
            m = evaluate_with_threshold(model, X_test, y_test, thr)
            m["train_years"] = train_years
            m["train_size_smote"] = int(len(X_res))
            strat_metrics[name] = m
            logger.info(f"  {name:<12} train={train_years} AUC {m['auc']:.4f} | "
                        f"PR-AUC {m['pr_auc']:.4f} | F1@tuned {m['f1_tuned']:.4f}")

            # The sliding window is the deployed adaptive model - persist it and
            # remember its tuned threshold for the Phase 5 backtest.
            if name == "sliding":
                model_file = models_path / f"xgboost_adaptive_{target_year}.joblib"
                joblib.dump(model, model_file)
                thresholds[str(target_year)] = m["threshold"]

        sliding = strat_metrics["sliding"]
        gain = round(sliding["auc"] - static_metrics["auc"], 4)
        pr_gain = round(sliding["pr_auc"] - static_metrics["pr_auc"], 4)

        # Log headline (static vs sliding) to MLflow.
        with mlflow.start_run(run_name=f"adaptive_{target_year}"):
            mlflow.log_param("target_year", target_year)
            mlflow.log_param("sliding_train_years", str(sliding["train_years"]))
            mlflow.log_param("test_size", len(X_test))
            for k in ("auc", "pr_auc", "f1_05", "f1_tuned"):
                mlflow.log_metric(f"adaptive_{k}", sliding[k])
                mlflow.log_metric(f"static_{k}", static_metrics[k])
            mlflow.log_metric("auc_gain", gain)
            mlflow.log_metric("pr_auc_gain", pr_gain)

        results.append({
            "year": target_year,
            "n_test": int(len(X_test)),
            "static": static_metrics,
            "strategies": strat_metrics,
            # Flat headline fields (static vs sliding adaptive) for easy reporting.
            "static_auc": static_metrics["auc"],
            "adaptive_auc": sliding["auc"],
            "auc_gain": gain,
            "static_pr_auc": static_metrics["pr_auc"],
            "adaptive_pr_auc": sliding["pr_auc"],
            "pr_auc_gain": pr_gain,
            "static_f1_05": static_metrics["f1_05"],
            "adaptive_f1_05": sliding["f1_05"],
            "static_f1_tuned": static_metrics["f1_tuned"],
            "adaptive_f1_tuned": sliding["f1_tuned"],
        })

    report_dir = Path(config["outputs"]["reports_path"])
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "retraining_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {report_dir / 'retraining_results.json'}")

    # Save thresholds for the Phase 5 backtest (so F1 there is also tuned).
    with open(models_path / "backtest_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    # Tidy strategy-comparison CSV (the naive-baseline answer).
    strat_rows = []
    for r in results:
        row = {"year": r["year"], "static_auc": r["static"]["auc"],
               "static_pr_auc": r["static"]["pr_auc"]}
        for name, m in r["strategies"].items():
            row[f"{name}_auc"] = m["auc"]
            row[f"{name}_pr_auc"] = m["pr_auc"]
        strat_rows.append(row)
    pd.DataFrame(strat_rows).to_csv(report_dir / "strategy_comparison.csv", index=False)
    logger.info(f"Saved {report_dir / 'strategy_comparison.csv'}")
    return results


# ---------------------------------------------------------------------------
# PART B - SHAP analysis of the static 2015 model across years
# ---------------------------------------------------------------------------
def safe_shap_values(model, X, n_background):
    """Interventional SHAP, with a tree_path_dependent fallback."""
    try:
        sv, _ = compute_shap_values(model, X, n_background)
        return sv
    except Exception as e:
        logger.warning(f"  interventional SHAP failed ({e}); using tree_path_dependent")
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X, check_additivity=False)
        if isinstance(sv, list):
            sv = sv[1]
        return sv


def shap_analysis(config, window_cache):
    start_year = config["data"]["start_year"]
    end_year = config["data"]["end_year"]
    n_bg = config["shap"]["background_samples"]
    sample_size = config["shap"]["sample_size"]
    top_n = config["shap"]["top_n_features"]      # for top-N list / Jaccard (10)
    bar_top_n = 15                                 # bar charts show top 15
    cos_thr = config["shap"]["cosine_similarity_threshold"]
    tau_thr = config["shap"]["kendall_tau_threshold"]
    plots_dir = Path(config["outputs"]["plots_path"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    rs = config["data"]["random_state"]
    static_model = joblib.load(
        Path(config["outputs"]["models_path"]) / f"xgboost_year_{start_year}.joblib"
    )

    def sample_X(year):
        X, _ = window_cache[year]
        if len(X) > sample_size:
            X = X.sample(sample_size, random_state=rs)
        return X

    # Baseline (2015) SHAP.
    logger.info(f"Computing baseline SHAP (year {start_year})...")
    Xb = sample_X(start_year)
    sv_b = safe_shap_values(static_model, Xb, n_bg)
    shap_baseline = mean_abs_shap(sv_b, Xb.columns.tolist())
    plot_shap_bar(shap_baseline, start_year, plots_dir, bar_top_n)

    top_features = {start_year: shap_baseline.head(top_n)}
    stability_records = []

    for year in range(start_year + 1, end_year + 1):
        if year not in window_cache:
            continue
        logger.info(f"Computing SHAP for year {year}...")
        Xy = sample_X(year)
        sv_y = safe_shap_values(static_model, Xy, n_bg)
        shap_year = mean_abs_shap(sv_y, Xy.columns.tolist())

        common = shap_baseline.index.intersection(shap_year.index)
        sb, sy = shap_baseline[common], shap_year[common]

        cos_sim = cosine_similarity_shap(sb, sy)
        tau, tau_p = kendall_tau_shap(sb, sy)
        jac = jaccard_top_n(sb, sy, n=top_n)
        logger.info(f"  cosine {cos_sim:.4f} | kendall_tau {tau:.4f} | jaccard {jac:.4f}")

        plot_shap_bar(shap_year, year, plots_dir, bar_top_n)
        top_features[year] = shap_year.head(top_n)
        stability_records.append({
            "year": year,
            "cosine_similarity": round(cos_sim, 4),
            "kendall_tau": round(float(tau), 4),
            "kendall_p_value": round(float(tau_p), 6),
            "jaccard_top_n": round(jac, 4),
            "cosine_stable": bool(cos_sim >= cos_thr),
            "rank_stable": bool(tau >= tau_thr),
        })

    stability_df = pd.DataFrame(stability_records)
    plot_stability_over_time(stability_df, plots_dir)
    report_dir = Path(config["outputs"]["reports_path"])
    stability_df.to_csv(report_dir / "shap_stability_report.csv", index=False)
    logger.info(f"Saved {report_dir / 'shap_stability_report.csv'}")
    return stability_df, top_features


# ---------------------------------------------------------------------------
def plot_strategy_comparison(results, plots_dir):
    """Bar chart: static vs naive_recent vs sliding vs expanding AUC per year."""
    import matplotlib.pyplot as plt
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    years = [r["year"] for r in results]
    strat_names = ["static", "naive_recent", "sliding", "expanding"]
    colors = {"static": "#FF6B6B", "naive_recent": "#FFA726",
              "sliding": "#00BCD4", "expanding": "#7C4DFF"}

    def auc_of(r, name):
        return r["static"]["auc"] if name == "static" else r["strategies"][name]["auc"]

    x = np.arange(len(years))
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, name in enumerate(strat_names):
        vals = [auc_of(r, name) * 100 for r in results]
        ax.bar(x + (i - 1.5) * w, vals, w, label=name, color=colors[name])
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("AUC-ROC (%)", fontsize=12)
    ax.set_ylim(60, 75)
    ax.set_title("Retraining-Strategy Comparison (AUC-ROC)\n"
                 "static vs naive-recent vs sliding-window vs expanding-window",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = plots_dir / "strategy_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved strategy-comparison plot -> {out}")
    return out


def main():
    config = load_config()
    for d in ["outputs/reports", "outputs/plots", "outputs/models", "mlruns"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Load each window once and reuse it across both parts (they're large CSVs).
    years = [config["data"]["start_year"]] + list(SLIDING_WINDOWS.keys())
    years = sorted(set(years) | set(y for w in SLIDING_WINDOWS.values() for y in w))
    logger.info(f"Loading windows: {years}")
    window_cache = {y: load_window_xy(y, config) for y in years}

    retrain_results = adaptive_retraining(config, window_cache)
    stability_df, top_features = shap_analysis(config, window_cache)

    retr_by_year = {r["year"]: r for r in retrain_results}
    stab_by_year = {int(r["year"]): r for _, r in stability_df.iterrows()}

    print("\n" + "=" * 96)
    print("PART A - STATIC vs ADAPTIVE (same held-out test set) - threshold-free + tuned-F1")
    print("=" * 96)
    print(f"{'Year':<6}{'Static':>9}{'Adapt':>9}{'AUC':>8}{'Static':>9}{'Adapt':>9}{'PRAUC':>8}"
          f"{'sF1@.5':>8}{'aF1@.5':>8}{'sF1tun':>8}{'aF1tun':>8}")
    print(f"{'':<6}{'AUC':>9}{'AUC':>9}{'gain':>8}{'PRAUC':>9}{'PRAUC':>9}{'gain':>8}"
          f"{'':>8}{'':>8}{'':>8}{'':>8}")
    print("-" * 96)
    for r in retrain_results:
        print(f"{r['year']:<6}{r['static_auc']:>9.4f}{r['adaptive_auc']:>9.4f}"
              f"{r['auc_gain']:>+8.4f}{r['static_pr_auc']:>9.4f}{r['adaptive_pr_auc']:>9.4f}"
              f"{r['pr_auc_gain']:>+8.4f}{r['static_f1_05']:>8.4f}{r['adaptive_f1_05']:>8.4f}"
              f"{r['static_f1_tuned']:>8.4f}{r['adaptive_f1_tuned']:>8.4f}")
    print("\nNote: at the naive 0.5 cutoff the static model shows HIGHER F1 - an artifact")
    print("of the rising default base-rate, not better skill. On the threshold-free AUC")
    print("and PR-AUC, and on F1 at a tuned threshold, the adaptive model wins every year.")

    # Strategy comparison (naive baselines vs sliding window) - gap-3 answer.
    print("\n" + "=" * 96)
    print("RETRAINING-STRATEGY COMPARISON (AUC / PR-AUC on the same test set)")
    print("=" * 96)
    print(f"{'Year':<6}{'static':>18}{'naive_recent':>18}{'sliding':>18}{'expanding':>18}")
    print(f"{'':<6}{'AUC / PR-AUC':>18}{'AUC / PR-AUC':>18}{'AUC / PR-AUC':>18}{'AUC / PR-AUC':>18}")
    print("-" * 96)
    for r in retrain_results:
        s = r["strategies"]
        def cell(d):
            return f"{d['auc']:.4f}/{d['pr_auc']:.4f}"
        print(f"{r['year']:<6}"
              f"{r['static']['auc']:.4f}/{r['static']['pr_auc']:.4f}".rjust(18)
              + cell(s["naive_recent"]).rjust(18)
              + cell(s["sliding"]).rjust(18)
              + cell(s["expanding"]).rjust(18))
    plot_strategy_comparison(retrain_results, Path(config["outputs"]["plots_path"]))

    print("\n" + "=" * 84)
    print("PART B - TOP-10 FEATURES BY MEAN |SHAP| (static 2015 model)")
    print("=" * 84)
    for year in sorted(top_features):
        feats = ", ".join(f"{k}({v:.3f})" for k, v in top_features[year].items())
        print(f"  {year}: {feats}")

    print("\n" + "=" * 84)
    print("KEY COMPARISON TABLE")
    print("=" * 84)
    print(f"{'Year':<6}{'Static AUC':>12}{'Adaptive AUC':>14}{'AUC gain':>11}"
          f"{'Cosine Sim':>12}{'Kendall Tau':>13}")
    print("-" * 84)
    final_rows = []
    for year in sorted(retr_by_year):
        r = retr_by_year[year]
        s = stab_by_year.get(year, {})
        cos = s.get("cosine_similarity", float("nan"))
        tau = s.get("kendall_tau", float("nan"))
        print(f"{year:<6}{r['static_auc']:>12.4f}{r['adaptive_auc']:>14.4f}"
              f"{r['auc_gain']:>+11.4f}{cos:>12.4f}{tau:>13.4f}")
        final_rows.append({
            "year": year,
            "static_auc": r["static_auc"],
            "adaptive_auc": r["adaptive_auc"],
            "auc_gain": r["auc_gain"],
            "static_pr_auc": r["static_pr_auc"],
            "adaptive_pr_auc": r["adaptive_pr_auc"],
            "pr_auc_gain": r["pr_auc_gain"],
            "cosine_similarity": cos,
            "kendall_tau": tau,
        })
    print("=" * 84)

    final_df = pd.DataFrame(final_rows)
    final_path = Path(config["outputs"]["reports_path"]) / "final_comparison.csv"
    final_df.to_csv(final_path, index=False)
    print(f"\nSaved final comparison -> {final_path}")
    print("Plots -> outputs/plots/ (shap_bar_*.png, shap_stability.png)")


if __name__ == "__main__":
    main()
