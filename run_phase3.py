"""
run_phase3.py
-------------
Phase 3 driver: Drift Detection.

  1. Runs the full PSI + KS drift pipeline (src/drift/detector.py) with
     window_2015 as reference, comparing 2016/2017/2018 against it.
     Saves outputs/reports/drift_report.json.
  2. Generates a PSI-over-time visualization -> outputs/plots/psi_over_time.png.
  3. Evaluates the STATIC 2015 XGBoost model on each year's data and reports
     how AUC-ROC degrades over time (the key research finding).
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score

from src.drift.detector import run_drift_detection

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "configs/config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def plot_psi_over_time(drift_log, config):
    """Plot mean PSI and the most-drifting features across years."""
    plots_dir = Path(config["outputs"]["plots_path"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    years = [e["year"] for e in drift_log]
    mean_psi = [e["mean_psi"] for e in drift_log]

    # Per-feature PSI across years; pick the features with the highest peak PSI.
    feats = list(drift_log[0]["psi_per_feature"].keys())
    peak = {f: max(e["psi_per_feature"].get(f, 0.0) for e in drift_log) for f in feats}
    top_feats = sorted(peak, key=peak.get, reverse=True)[:6]

    warn = config["drift"]["psi_threshold_warning"]
    crit = config["drift"]["psi_threshold_critical"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for f in top_feats:
        series = [e["psi_per_feature"].get(f, 0.0) for e in drift_log]
        ax.plot(years, series, marker="o", linewidth=1.8, alpha=0.85, label=f)

    ax.plot(years, mean_psi, marker="D", color="black", linewidth=3.0,
            label="MEAN PSI (all features)")

    ax.axhline(warn, color="orange", linestyle="--", linewidth=1.5,
               label=f"Warning ({warn})")
    ax.axhline(crit, color="red", linestyle="--", linewidth=1.5,
               label=f"Critical / retrain ({crit})")

    ax.set_xlabel("Year (vs 2015 reference)", fontsize=12)
    ax.set_ylabel("PSI", fontsize=12)
    ax.set_title("Population Stability Index Over Time\n(feature drift vs 2015 baseline)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(years)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = plots_dir / "psi_over_time.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved PSI plot -> {out_path}")
    return out_path


def plot_drift_methods(drift_log, config):
    """Plot the three drift signals (PSI, JS, classifier-AUC) on a shared timeline."""
    plots_dir = Path(config["outputs"]["plots_path"])
    plots_dir.mkdir(parents=True, exist_ok=True)
    years = [e["year"] for e in drift_log]

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(years, [e["mean_psi"] for e in drift_log], marker="o",
             color="#D4AC0D", linewidth=2.4, label="Mean PSI (unbounded)")
    ax1.plot(years, [e["mean_js_distance"] for e in drift_log], marker="s",
             color="#7C4DFF", linewidth=2.4, label="Mean JS distance [0,1]")
    ax1.set_xlabel("Year (vs 2015 reference)", fontsize=12)
    ax1.set_ylabel("Per-feature drift (PSI / JS)", fontsize=12)
    ax1.set_xticks(years)

    ax2 = ax1.twinx()
    ax2.plot(years, [e["classifier_drift_auc"] for e in drift_log], marker="D",
             color="#00BCD4", linewidth=2.8, label="Classifier two-sample AUC")
    ax2.axhline(0.5, color="grey", linestyle=":", linewidth=1.3, label="C2ST no-drift (0.5)")
    ax2.set_ylabel("Classifier two-sample test AUC", fontsize=12)
    ax2.set_ylim(0.45, 1.0)

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=9, loc="upper left")
    ax1.set_title("Drift Detected by Three Independent Methods Over Time",
                  fontsize=14, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    plt.tight_layout()
    out_path = plots_dir / "drift_methods_over_time.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved drift-methods plot -> {out_path}")
    return out_path


def save_drift_methods_table(drift_log, config):
    """Write a tidy per-year, per-method drift-comparison CSV + trigger policies."""
    rows = []
    for e in drift_log:
        rows.append({
            "year": e["year"],
            "mean_psi": e["mean_psi"],
            "psi_critical_features": e["n_flagged_features"],
            "ks_label_drift": e["label_drift"],
            "mean_js_distance": e["mean_js_distance"],
            "classifier_drift_auc": e["classifier_drift_auc"],
            "classifier_drift_detected": e["classifier_drift_detected"],
            # Trigger policies - what WOULD fire under each rule.
            "trigger_psi_only": e["psi_drift"],
            "trigger_label_only": e["label_drift"],
            "trigger_classifier_only": e["classifier_drift_detected"],
            "trigger_any": e["retrain_triggered"],
        })
    df = pd.DataFrame(rows)
    out = Path(config["outputs"]["reports_path"]) / "drift_methods_comparison.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved drift-methods table -> {out}")
    return df


def evaluate_static_model_degradation(config):
    """Evaluate the static 2015 XGBoost model on every year's data."""
    models_path = Path(config["outputs"]["models_path"])
    windows_path = Path(config["data"]["windows_path"])
    start_year = config["data"]["start_year"]
    end_year = config["data"]["end_year"]

    static_path = models_path / f"xgboost_year_{start_year}.joblib"
    static_model = joblib.load(static_path)
    logger.info(f"Loaded static model: {static_path}")

    rows = []
    baseline_auc = None
    for year in range(start_year, end_year + 1):
        wpath = windows_path / f"window_{year}.csv"
        if not wpath.exists():
            continue
        df = pd.read_csv(wpath)
        X = df.drop(columns=["target", "year"], errors="ignore")
        y = df["target"]
        proba = static_model.predict_proba(X)[:, 1]
        pred = static_model.predict(X)
        auc = roc_auc_score(y, proba)
        f1 = f1_score(y, pred)
        if baseline_auc is None:
            baseline_auc = auc
        rows.append({
            "year": year,
            "n": len(y),
            "default_rate": round(float(y.mean()), 4),
            "static_auc": round(float(auc), 4),
            "static_f1": round(float(f1), 4),
            "auc_drop_vs_2015": round(float(baseline_auc - auc), 4),
            "auc_drop_pct": round(float((baseline_auc - auc) / baseline_auc * 100), 2),
        })
    return pd.DataFrame(rows)


def main():
    config = load_config()
    for d in ["outputs/reports", "outputs/plots"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    drift_log, retrain_years = run_drift_detection(CONFIG_PATH)

    plot_path = plot_psi_over_time(drift_log, config)
    plot_drift_methods(drift_log, config)
    save_drift_methods_table(drift_log, config)

    deg = evaluate_static_model_degradation(config)
    deg_path = Path(config["outputs"]["reports_path"]) / "static_degradation.csv"
    deg.to_csv(deg_path, index=False)

    # ---- Final summary ----
    print("\n" + "=" * 78)
    print("PHASE 3 SUMMARY - DRIFT DETECTION (reference year = 2015)")
    print("=" * 78)
    print(f"{'Year':<6}{'MeanPSI':>9}{'#Flagged(>0.2)':>16}{'KS stat':>10}"
          f"{'KS p':>12}{'LabelDrift':>12}{'Retrain':>10}")
    print("-" * 78)
    for e in drift_log:
        print(f"{e['year']:<6}{e['mean_psi']:>9.4f}{e['n_flagged_features']:>16}"
              f"{e['ks_statistic']:>10.4f}{e['ks_p_value']:>12.2e}"
              f"{('YES' if e['label_drift'] else 'no'):>12}"
              f"{('YES' if e['retrain_triggered'] else 'no'):>10}")
    print("-" * 78)
    print(f"Years requiring retraining: {retrain_years}")

    print("\n" + "=" * 78)
    print("MULTI-METHOD DRIFT DETECTION (PSI + KS + JS + classifier two-sample test)")
    print("=" * 78)
    print(f"{'Year':<6}{'MeanPSI':>9}{'MeanJS':>9}{'C2ST AUC':>10}{'C2ST?':>8}"
          f"{'PSI?':>7}{'Label?':>8}")
    print("-" * 78)
    for e in drift_log:
        print(f"{e['year']:<6}{e['mean_psi']:>9.4f}{e['mean_js_distance']:>9.4f}"
              f"{e['classifier_drift_auc']:>10.4f}"
              f"{('YES' if e['classifier_drift_detected'] else 'no'):>8}"
              f"{('YES' if e['psi_drift'] else 'no'):>7}"
              f"{('YES' if e['label_drift'] else 'no'):>8}")
    print("-" * 78)
    print("Trigger-policy comparison - years each policy would retrain:")
    pol = {
        "PSI-only      ": [e["year"] for e in drift_log if e["psi_drift"]],
        "Label-only (KS)": [e["year"] for e in drift_log if e["label_drift"]],
        "Classifier-only": [e["year"] for e in drift_log if e["classifier_drift_detected"]],
        "ANY (deployed)  ": [e["year"] for e in drift_log if e["retrain_triggered"]],
    }
    for name, yrs in pol.items():
        print(f"  {name}: {yrs}")
    print("Note: drift fires every year under the deployed 'ANY' policy, so drift-")
    print("triggered retraining == periodic yearly retraining ON THIS DATA. The")
    print("trigger's value is operational: a PSI-only policy would SKIP 2016 (no")
    print("critical covariate PSI), avoiding an unnecessary retrain when only the")
    print("label prior - not the feature distribution - had moved.")
    for e in drift_log:
        tops = ", ".join(f"{d['feature']}({d['separation_auc']:.3f})"
                         for d in e["classifier_top_features"][:5])
        print(f"  {e['year']} classifier-top drift features: {tops}")

    print("\nFlagged features per year (PSI > 0.2):")
    for e in drift_log:
        print(f"  {e['year']}: {e['flagged_features'] if e['flagged_features'] else '(none)'}")

    print("\nTop feature PSI per year:")
    for e in drift_log:
        top = sorted(e["psi_per_feature"].items(), key=lambda kv: kv[1], reverse=True)[:5]
        print(f"  {e['year']}: " + ", ".join(f"{k}={v:.3f}" for k, v in top))

    print("\n" + "=" * 78)
    print("KEY FINDING - STATIC 2015 MODEL AUC DEGRADATION OVER TIME")
    print("=" * 78)
    print(f"{'Year':<6}{'N':>9}{'DefaultRate':>13}{'AUC':>9}{'F1':>9}"
          f"{'AUC drop':>11}{'drop %':>9}")
    print("-" * 78)
    for _, r in deg.iterrows():
        print(f"{int(r['year']):<6}{int(r['n']):>9}{r['default_rate']:>13.4f}"
              f"{r['static_auc']:>9.4f}{r['static_f1']:>9.4f}"
              f"{r['auc_drop_vs_2015']:>11.4f}{r['auc_drop_pct']:>8.2f}%")
    print("=" * 78)
    print(f"\nReports : {Path(config['outputs']['reports_path']) / 'drift_report.json'}")
    print(f"          {deg_path}")
    print(f"Plot    : {plot_path}")


if __name__ == "__main__":
    main()
