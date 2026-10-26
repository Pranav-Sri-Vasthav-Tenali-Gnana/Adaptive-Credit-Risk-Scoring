"""
data_access.py
--------------
Loading + live-inference helpers for the Streamlit dashboard. Kept separate from
the UI so the (cached) data logic is testable on its own.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import joblib

# Make `src` importable when Streamlit runs the app from the project root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.drift.psi import calculate_psi_all_features            # noqa: E402
from src.drift.ks_test import ks_test_labels                    # noqa: E402
from src.drift.advanced_detectors import (                      # noqa: E402
    js_distance_all_features, classifier_drift_test, classifier_drift_top_features,
)
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score  # noqa: E402

REPORTS = ROOT / "outputs" / "reports"
PLOTS = ROOT / "outputs" / "plots"
MODELS = ROOT / "outputs" / "models"
WINDOWS = ROOT / "data" / "windows"

NUMERICAL = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc",
    "pub_rec_bankruptcies",
]
REFERENCE_YEAR = 2015


# artifact loaders
def read_csv(name):
    p = REPORTS / name
    return pd.read_csv(p) if p.exists() else None


def read_json(name):
    p = REPORTS / name
    return json.loads(p.read_text()) if p.exists() else None


def read_text(name):
    p = REPORTS / name
    return p.read_text(encoding="utf-8") if p.exists() else None


def plot_path(name):
    p = PLOTS / name
    return str(p) if p.exists() else None


def load_thresholds():
    p = MODELS / "backtest_thresholds.json"
    return json.loads(p.read_text()) if p.exists() else {}


def feature_columns(reference_df):
    return [c for c in reference_df.columns if c not in ("target", "year")]


def load_window(year):
    p = WINDOWS / f"window_{year}.csv"
    return pd.read_csv(p) if p.exists() else None


def available_window_years():
    return sorted(int(p.stem.split("_")[1]) for p in WINDOWS.glob("window_*.csv"))


def available_adaptive_years():
    return sorted(int(p.stem.split("_")[-1])
                  for p in MODELS.glob("xgboost_adaptive_*.joblib"))


# live drift + scoring
def align_features(df, ref_feature_cols):
    """Reindex an incoming frame onto the reference feature columns (one-hot
    categories absent in the upload become 0), so saved models can score it."""
    X = df.drop(columns=[c for c in ("target", "year") if c in df.columns],
                errors="ignore")
    missing = [c for c in ref_feature_cols if c not in X.columns]
    extra = [c for c in X.columns if c not in ref_feature_cols]
    X = X.reindex(columns=ref_feature_cols, fill_value=0).astype(np.float32)
    return X, missing, extra


def run_live_drift(reference_df, current_df, psi_critical=0.2, ks_alpha=0.05):
    """Run PSI + KS + JS + classifier two-sample test of current vs reference."""
    feat_cols = feature_columns(reference_df)

    psi = calculate_psi_all_features(reference_df, current_df, NUMERICAL)
    flagged = [f for f, v in psi.items() if v >= psi_critical]
    mean_psi = float(np.mean(list(psi.values()))) if psi else 0.0

    js = js_distance_all_features(reference_df, current_df, NUMERICAL)
    mean_js = float(np.mean(list(js.values()))) if js else 0.0

    label = None
    if "target" in current_df.columns:
        label = ks_test_labels(reference_df["target"], current_df["target"], ks_alpha)

    c2st = classifier_drift_test(reference_df, current_df, feat_cols)
    c2st_top = classifier_drift_top_features(reference_df, current_df, feat_cols)

    label_drift = bool(label and label["drift_detected"])
    retrain = (len(flagged) > 0) or c2st["drift_detected"] or label_drift
    return {
        "psi": psi, "mean_psi": round(mean_psi, 4), "flagged": flagged,
        "js": js, "mean_js": round(mean_js, 4),
        "label": label,
        "c2st": c2st, "c2st_top": c2st_top,
        "retrain": bool(retrain),
    }


def score_models(current_df, reference_df, adaptive_year, holdout=False):
    """Score the current data with the static-2015 model and a chosen adaptive
    model. Returns per-model probabilities and (if labels present) AUC/PR-AUC/F1.

    holdout=True reconstructs the SAME held-out 20% test split used in Phase 4/5
    (test_size=0.2, random_state=42, stratified) and scores only on it. Use this
    for built-in windows so the adaptive model is evaluated leakage-free (it was
    trained on the other 80%) and the numbers match the thesis exactly. For a
    genuinely new uploaded file there is no leakage, so score the whole thing.
    """
    ref_cols = feature_columns(reference_df)
    df = current_df
    eval_scope = "full data (new / uploaded)"
    if holdout and "target" in current_df.columns:
        from sklearn.model_selection import train_test_split
        _, df = train_test_split(current_df, test_size=0.2, random_state=42,
                                 stratify=current_df["target"])
        eval_scope = "held-out 20% test split (leakage-free, matches thesis)"
    X, missing, extra = align_features(df, ref_cols)
    y = df["target"].astype(int).values if "target" in df.columns else None
    thr = load_thresholds()

    out = {"n": int(len(X)), "missing_cols": missing, "extra_cols": extra,
           "has_labels": y is not None, "eval_scope": eval_scope, "models": {}}

    static = joblib.load(MODELS / f"xgboost_year_{REFERENCE_YEAR}.joblib")
    specs = [("Static (2015, never retrained)", static, thr.get("static", 0.5))]
    apath = MODELS / f"xgboost_adaptive_{adaptive_year}.joblib"
    if apath.exists():
        specs.append((f"Adaptive (retrained, {adaptive_year})",
                      joblib.load(apath), thr.get(str(adaptive_year), 0.5)))

    for name, model, t in specs:
        proba = model.predict_proba(X)[:, 1]
        entry = {"proba": proba, "threshold": float(t),
                 "pred_rate": float((proba >= t).mean())}
        if y is not None:
            entry.update({
                "auc": round(float(roc_auc_score(y, proba)), 4),
                "pr_auc": round(float(average_precision_score(y, proba)), 4),
                "f1_tuned": round(float(f1_score(y, (proba >= t).astype(int))), 4),
            })
        out["models"][name] = entry
    return out
