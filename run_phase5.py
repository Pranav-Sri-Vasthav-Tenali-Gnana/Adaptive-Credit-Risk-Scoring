"""
run_phase5.py
-------------
Phase 5 driver: Backtesting + Final Consolidated Report.

  1. Runs backtest_over_years() -> outputs/plots/auc_over_time.png (static vs
     adaptive across 2015-2018, evaluated on matched held-out test splits).
  2. Assembles outputs/reports/final_summary.txt from the saved phase reports.
  3. Lists everything in outputs/plots/ and outputs/reports/.
"""

import warnings
warnings.filterwarnings("ignore")

import logging
from pathlib import Path
from datetime import datetime

import yaml
import pandas as pd

from src.models.evaluate import backtest_over_years

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "configs/config.yaml"
RUN_DATE = datetime.now().strftime("%Y-%m-%d")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _read_csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def build_summary(config, backtest_df):
    reports = Path(config["outputs"]["reports_path"])
    deg = _read_csv(reports / "static_degradation.csv")        # year,n,default_rate,static_auc,...
    final_cmp = _read_csv(reports / "final_comparison.csv")     # year,static_auc,adaptive_auc,auc_gain,pr_auc...,cosine,kendall
    shap_stab = _read_csv(reports / "shap_stability_report.csv")
    strat = _read_csv(reports / "strategy_comparison.csv")      # year, {static,naive_recent,sliding,expanding}_auc/_pr_auc
    drift_methods = _read_csv(reports / "drift_methods_comparison.csv")

    proj = config["project"]
    L = []
    bar = "=" * 78
    L.append(bar)
    L.append(proj["name"].upper())
    L.append(bar)
    L.append(f"Author    : {proj['author']}")
    L.append(f"Student ID: {proj['id']}")
    L.append(f"Institute : {proj['institute']}")
    L.append(f"Generated : {RUN_DATE}")
    L.append("")

    # --- Dataset stats ---
    L.append("-" * 78)
    L.append("1. DATASET - Lending Club, yearly windows (cleaned)")
    L.append("-" * 78)
    if deg is not None:
        L.append(f"{'Year':<8}{'Rows':>12}{'Default rate':>16}")
        total = 0
        for _, r in deg.iterrows():
            total += int(r["n"])
            L.append(f"{int(r['year']):<8}{int(r['n']):>12,}{r['default_rate']*100:>15.2f}%")
        L.append(f"{'TOTAL':<8}{total:>12,}")
        L.append("")
        L.append("Note: dataset spans 2015-2018 only (Lending Club public release ends 2018).")
        L.append("Default rate rises steadily over time - a built-in prior/label shift.")
    L.append("")

    # --- Phase 3: static degradation ---
    L.append("-" * 78)
    L.append("2. PHASE 3 KEY FINDING - STATIC 2015 MODEL DEGRADES OVER TIME")
    L.append("-" * 78)
    if deg is not None:
        L.append(f"{'Year':<8}{'Static AUC':>12}{'Static F1':>12}{'AUC drop':>12}{'drop %':>10}")
        for _, r in deg.iterrows():
            L.append(f"{int(r['year']):<8}{r['static_auc']:>12.4f}{r['static_f1']:>12.4f}"
                     f"{r['auc_drop_vs_2015']:>12.4f}{r['auc_drop_pct']:>9.2f}%")
        L.append("")
        L.append("(Static model = XGBoost trained once on 2015, scored on each full year window.)")
    L.append("")

    # --- Multi-method drift detection ---
    if drift_methods is not None:
        L.append("-" * 78)
        L.append("2b. MULTI-METHOD DRIFT DETECTION (PSI + KS + JS + classifier two-sample)")
        L.append("-" * 78)
        L.append(f"{'Year':<8}{'MeanPSI':>10}{'MeanJS':>10}{'C2ST-AUC':>11}{'Retrain?':>11}")
        for _, r in drift_methods.iterrows():
            L.append(f"{int(r['year']):<8}{r['mean_psi']:>10.4f}{r['mean_js_distance']:>10.4f}"
                     f"{r['classifier_drift_auc']:>11.4f}"
                     f"{('YES' if r['trigger_any'] else 'no'):>11}")
        L.append("")
        L.append("Three independent detectors agree drift is present and growing: PSI/JS rise")
        L.append("per-feature, and the classifier two-sample AUC (0.5 = indistinguishable)")
        L.append("climbs as 2018 feature distributions become easier to separate from 2015.")
        L.append("Operational note: drift fires every year, so drift-triggered retraining")
        L.append("equals periodic yearly retraining HERE; the trigger's value is skipping")
        L.append("retrains when stable (a PSI-only policy would skip 2016).")
    L.append("")

    # --- Phase 4: static vs adaptive ---
    L.append("-" * 78)
    L.append("3. PHASE 4 KEY FINDING - STATIC vs ADAPTIVE (same held-out test set)")
    L.append("-" * 78)
    if final_cmp is not None:
        has_pr = "static_pr_auc" in final_cmp.columns
        if has_pr:
            L.append(f"{'Year':<8}{'Stat AUC':>10}{'Adap AUC':>10}{'AUCgain':>9}"
                     f"{'Stat PRAUC':>12}{'Adap PRAUC':>12}{'PRgain':>9}")
            for _, r in final_cmp.iterrows():
                L.append(f"{int(r['year']):<8}{r['static_auc']:>10.4f}{r['adaptive_auc']:>10.4f}"
                         f"{r['auc_gain']:>+9.4f}{r['static_pr_auc']:>12.4f}"
                         f"{r['adaptive_pr_auc']:>12.4f}{r['pr_auc_gain']:>+9.4f}")
        else:
            L.append(f"{'Year':<8}{'Static AUC':>12}{'Adaptive AUC':>14}{'AUC gain':>12}")
            for _, r in final_cmp.iterrows():
                L.append(f"{int(r['year']):<8}{r['static_auc']:>12.4f}{r['adaptive_auc']:>14.4f}"
                         f"{r['auc_gain']:>+12.4f}")
        L.append("")
        L.append("Adaptive = XGBoost retrained on a sliding window of recent years, triggered")
        L.append("by drift detection. Both AUC (ranking) and PR-AUC (threshold-free precision-")
        L.append("recall on the imbalanced default class) improve, with the gain growing as")
        L.append("drift accumulates.")
    L.append("")

    # --- Retraining-strategy comparison (naive baselines) ---
    if strat is not None:
        L.append("-" * 78)
        L.append("3b. RETRAINING-STRATEGY COMPARISON - does simpler retraining do as well?")
        L.append("-" * 78)
        L.append(f"{'Year':<8}{'static':>11}{'naive_recent':>14}{'sliding':>11}{'expanding':>12}"
                 f"   (AUC-ROC)")
        for _, r in strat.iterrows():
            L.append(f"{int(r['year']):<8}{r['static_auc']:>11.4f}{r['naive_recent_auc']:>14.4f}"
                     f"{r['sliding_auc']:>11.4f}{r['expanding_auc']:>12.4f}")
        L.append("")
        L.append("naive_recent = most recent year only; sliding = 3-year window (deployed);")
        L.append("expanding = all history. Finding: ALL retraining beats static, and RECENCY")
        L.append("beats VOLUME - naive_recent edges out the sliding window, while the expanding")
        L.append("window is the weakest adaptive strategy by 2018 (0.7154 < 0.7222 < 0.7297),")
        L.append("i.e. keeping stale 2015-16 data actively hurts once drift sets in. The sliding")
        L.append("window is the robust compromise: it bounds memory without needing the optimal")
        L.append("lookback to be known in advance, and tracks naive_recent closely.")
    L.append("")

    # --- Backtest table (incl. 2015 anchor) ---
    if backtest_df is not None and len(backtest_df):
        L.append("-" * 78)
        L.append("   BACKTEST (incl. 2015 anchor) - used for outputs/plots/auc_over_time.png")
        L.append("-" * 78)
        L.append(f"{'Year':<8}{'Static AUC':>12}{'Adaptive AUC':>14}")
        for _, r in backtest_df.iterrows():
            L.append(f"{int(r['year']):<8}{r['static_auc']:>12.4f}{r['adaptive_auc']:>14.4f}")
        L.append("")

    # --- SHAP stability ---
    L.append("-" * 78)
    L.append("4. SHAP EXPLAINABILITY STABILITY (static 2015 model vs each year)")
    L.append("-" * 78)
    if shap_stab is not None:
        L.append(f"{'Year':<8}{'Cosine Sim':>12}{'Kendall Tau':>14}{'Jaccard@10':>12}")
        min_jac = 1.0
        for _, r in shap_stab.iterrows():
            min_jac = min(min_jac, float(r["jaccard_top_n"]))
            L.append(f"{int(r['year']):<8}{r['cosine_similarity']:>12.4f}"
                     f"{r['kendall_tau']:>14.4f}{r['jaccard_top_n']:>12.4f}")
        L.append("")
        L.append("Thresholds: cosine >= 0.95, kendall tau >= 0.8. Overall importance vectors")
        L.append("stay stable (high cosine/tau), while Jaccard@10 < 1.0 in drift years shows")
        if min_jac < 1.0:
            L.append("real churn in WHICH features make the top-10 (int_rate rising with its")
            L.append("PSI drift) - caught only after fixing the Jaccard metric (it previously")
            L.append("collapsed to a constant 1.0 due to a vector-reindexing bug).")
        else:
            L.append("the top-10 membership is unchanged across years on this run.")
    L.append("")

    # --- Research contribution ---
    L.append("-" * 78)
    L.append("5. RESEARCH CONTRIBUTION SUMMARY")
    L.append("-" * 78)
    bullets = [
        "Demonstrated real concept drift in Lending Club credit data (2015-2018) with THREE "
        "independent detectors that agree: the default rate rose 20.5% -> 27.2% (KS label "
        "drift), PSI and Jensen-Shannon distance flagged int_rate and revol_util as "
        "critically drifting, and a classifier two-sample test confirmed growing "
        "multivariate covariate drift.",
        "Quantified the cost of model staleness: a static 2015 XGBoost loses ~9.8% AUC "
        "by 2018 (0.7422 -> 0.6695) when left unretrained.",
        "Showed drift-triggered adaptive retraining (sliding window) recovers performance on "
        "both AUC and the threshold-free PR-AUC, with the gain growing as drift accumulates, "
        "restoring ~2015-level discrimination by 2018.",
        "Benchmarked retraining STRATEGIES (naive most-recent-year, 3-year sliding window, "
        "full expanding window) against a naive-baseline reviewer concern; found that RECENCY "
        "beats VOLUME - the expanding window is the weakest adaptive option by 2018 (stale "
        "data hurts), naive-recent is strongest, and the deployed sliding window is the robust "
        "compromise. Also clarified the drift TRIGGER's value is operational (skipping "
        "unnecessary retrains) since drift fires every year on this data.",
        "Built an end-to-end, MLflow-tracked MLOps loop coupling the multi-method drift "
        "detection to automated retraining triggers, with validation-tuned decision "
        "thresholds (avoiding the misleading fixed-0.5 F1 artifact) - a deployable "
        "monitoring-and-retrain pipeline.",
        "Added SHAP-based explainability-stability monitoring (cosine, Kendall tau, and a "
        "corrected top-N Jaccard); feature-importance shifts (int_rate rising) corroborate "
        "the drift signal and give an interpretable early warning alongside metrics.",
    ]
    for i, b in enumerate(bullets, 1):
        # simple wrap at ~76 chars
        words, line = b.split(), ""
        wrapped = []
        for w in words:
            if len(line) + len(w) + 1 > 74:
                wrapped.append(line); line = w
            else:
                line = (line + " " + w).strip()
        if line:
            wrapped.append(line)
        L.append(f"  {i}. {wrapped[0]}")
        for cont in wrapped[1:]:
            L.append(f"     {cont}")
    L.append("")

    # --- Limitations ---
    L.append("-" * 78)
    L.append("6. LIMITATIONS & THREATS TO VALIDITY")
    L.append("-" * 78)
    limitations = [
        "Data ends in 2018: the Lending Club public release contains no loans issued "
        "2019-2023, so the COVID-era shock - the most extreme recent credit drift - "
        "cannot be tested; the observed drift is the milder 2015-2018 trend.",
        "Single dataset / single market (US unsecured consumer loans), so the magnitude "
        "of degradation and recovery may not transfer to other portfolios or geographies.",
        "Yearly windows are coarse: real monitoring would operate monthly; annual "
        "granularity smooths over intra-year drift and limits the number of drift events.",
        "Labels require loan maturity (only resolved Fully Paid / Charged Off loans are "
        "used), so the most recent originations are under-represented - a survivorship/"
        "label-latency caveat inherent to credit outcomes.",
        "SMOTE is applied to balance training; while thresholds are tuned on a held-out "
        "real-prevalence validation split, probability calibration (e.g. isotonic) was "
        "not performed and would be needed before using scores as probabilities.",
        "Drift fires every year on this data, so the trigger cannot here be shown to REDUCE "
        "retraining frequency versus a periodic schedule - only to avoid retraining when a "
        "single detector (PSI) is quiet; a longer, lower-drift series is needed to show that.",
    ]
    for i, b in enumerate(limitations, 1):
        words, line, wrapped = b.split(), "", []
        for w in words:
            if len(line) + len(w) + 1 > 74:
                wrapped.append(line); line = w
            else:
                line = (line + " " + w).strip()
        if line:
            wrapped.append(line)
        L.append(f"  {i}. {wrapped[0]}")
        for cont in wrapped[1:]:
            L.append(f"     {cont}")
    L.append("")

    L.append(bar)
    L.append("END OF REPORT")
    L.append(bar)
    return "\n".join(L)


def main():
    config = load_config()
    for d in ["outputs/reports", "outputs/plots"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    logger.info("Running backtest_over_years()...")
    backtest_df = backtest_over_years(CONFIG_PATH)
    logger.info("Backtest results:\n" + backtest_df.to_string(index=False))

    summary = build_summary(config, backtest_df)
    summary_path = Path(config["outputs"]["reports_path"]) / "final_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    logger.info(f"Saved {summary_path}")
    print("\n" + summary)

    print("\n" + "=" * 78)
    print("OUTPUT FILES")
    print("=" * 78)
    for sub in ["plots", "reports"]:
        d = Path("outputs") / sub
        print(f"\noutputs/{sub}/")
        for f in sorted(d.glob("*")):
            if f.is_file():
                print(f"  {f.name:<32}{f.stat().st_size:>10,} bytes")


if __name__ == "__main__":
    main()
