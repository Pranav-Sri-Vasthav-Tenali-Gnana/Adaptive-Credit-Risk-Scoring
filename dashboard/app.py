"""
app.py - Streamlit dashboard for the Adaptive Credit Risk Scoring project.

Run from the project root:
    venv\\Scripts\\python.exe -m streamlit run dashboard/app.py

Pages:
  1. Overview                 - headline story + dataset
  2. Drift Detection          - PSI + KS + JS + classifier two-sample test
  3. Static vs Adaptive       - AUC / PR-AUC / tuned-F1 over time
  4. Retraining Strategies    - static vs naive-recent vs sliding vs expanding
  5. Explainability (SHAP)    - importance bars + stability metrics
  6. Live Drift Check         - upload / pick a window, detect drift + score LIVE
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dashboard.data_access as da

ACCENT = {"static": "#FF6B6B", "adaptive": "#00BCD4", "violet": "#7C4DFF",
          "gold": "#D4AC0D", "orange": "#FFA726"}

st.set_page_config(page_title="Adaptive Credit Risk Scoring",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .block-container {padding-top: 2rem;}
  div[data-testid="stMetricValue"] {font-size: 1.6rem;}
  .small {color: #888; font-size: 0.85rem;}
</style>
""", unsafe_allow_html=True)


# cached loaders
@st.cache_data(show_spinner=False)
def get_csv(name): return da.read_csv(name)

@st.cache_data(show_spinner=False)
def get_json(name): return da.read_json(name)

@st.cache_data(show_spinner=False)
def get_text(name): return da.read_text(name)

@st.cache_data(show_spinner=False)
def get_window(year): return da.load_window(year)

@st.cache_resource(show_spinner=False)
def get_reference():
    return da.load_window(da.REFERENCE_YEAR)


def fmt_pct(x): return f"{x*100:.2f}%"


def page_overview():
    st.title("Adaptive Credit Risk Scoring with Concept Drift Detection")
    st.caption("M.Tech Data Engineering, IIT Jodhpur, Pranav Sri Vasthav Tenali "
               "Gnana (M25DE1069)")

    deg = get_csv("static_degradation.csv")
    cmp = get_csv("final_comparison.csv")
    drift = get_csv("drift_methods_comparison.csv")

    st.markdown("#### The problem in one line")
    st.info("A credit-risk model trained once goes stale: as borrower behaviour and "
            "the lending market drift, a static 2015 model's discrimination decays "
            "year over year. This project **detects** that drift with several "
            "independent methods and **adapts** by retraining - recovering the lost "
            "performance.")

    if deg is not None and cmp is not None:
        last = deg.iloc[-1]
        c2018 = cmp[cmp["year"] == 2018].iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Static AUC drop by 2018", f"-{last['auc_drop_pct']:.1f}%",
                  help="Static 2015 XGBoost AUC: 0.7422 -> 0.6695 on 2018 data.")
        k2.metric("Adaptive AUC gain, 2018", f"+{c2018['auc_gain']:.4f}",
                  help="Sliding-window adaptive model vs static, same test set.")
        k3.metric("Adaptive PR-AUC gain, 2018", f"+{c2018['pr_auc_gain']:.4f}",
                  help="Threshold-free metric for the imbalanced default class.")
        if drift is not None:
            k4.metric("Drift detectors agreeing", "3 / 3",
                      help="PSI, KS label test, Jensen-Shannon, and a classifier "
                           "two-sample test all flag growing drift.")

    st.markdown("#### Dataset - Lending Club, yearly windows (cleaned)")
    if deg is not None:
        d = deg[["year", "n", "default_rate"]].copy()
        d["default_rate"] = (d["default_rate"] * 100).round(2).astype(str) + "%"
        d.columns = ["Year", "Rows", "Default rate"]
        d["Rows"] = d["Rows"].map(lambda v: f"{int(v):,}")
        st.dataframe(d, hide_index=True, use_container_width=True)
        st.caption("Default rate rises 20.5% -> 27.2% over 2015-2018 - a built-in "
                   "label/prior shift. Public release ends in 2018 (no 2019-23 loans).")

    with st.expander("Full consolidated text report (final_summary.txt)"):
        txt = get_text("final_summary.txt")
        st.text(txt or "final_summary.txt not found - run run_phase5.py.")


def page_drift():
    st.title("Drift Detection - four independent methods")
    dm = get_csv("drift_methods_comparison.csv")
    report = get_json("drift_report.json")
    if dm is None:
        st.warning("Run run_phase3.py first.")
        return

    st.markdown("Per-feature **PSI** and **Jensen-Shannon distance**, a **KS** test on "
                "the label, and a **classifier two-sample test** (can a model tell the "
                "year apart? 0.5 = no, 1.0 = perfectly) - all vs the 2015 reference.")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=dm["year"], y=dm["mean_psi"], name="Mean PSI",
                  mode="lines+markers", line=dict(color=ACCENT["gold"], width=3)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=dm["year"], y=dm["mean_js_distance"], name="Mean JS distance",
                  mode="lines+markers", line=dict(color=ACCENT["violet"], width=3)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=dm["year"], y=dm["classifier_drift_auc"],
                  name="Classifier 2-sample AUC", mode="lines+markers",
                  line=dict(color=ACCENT["adaptive"], width=3, dash="dot")),
                  secondary_y=True)
    fig.add_hline(y=0.5, line=dict(color="grey", dash="dot"), secondary_y=True)
    fig.update_xaxes(title_text="Year (vs 2015)", tickmode="array",
                     tickvals=dm["year"].tolist())
    fig.update_yaxes(title_text="PSI / JS (per-feature)", secondary_y=False)
    fig.update_yaxes(title_text="Classifier AUC", range=[0.45, 1.02], secondary_y=True)
    fig.update_layout(height=440, legend=dict(orientation="h", y=1.12),
                      margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.success("**Key insight:** the classifier two-sample AUC is ~0.98+ even though "
               "the most drifted *single* feature only reaches ~0.65 univariate AUC - "
               "the drift is **multivariate**, which per-feature PSI (0-2 features "
               "flagged) badly understates.")

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("##### Detector readings per year")
        show = dm[["year", "mean_psi", "psi_critical_features", "ks_label_drift",
                   "mean_js_distance", "classifier_drift_auc", "trigger_any"]].copy()
        show.columns = ["Year", "Mean PSI", "PSI>=0.2", "KS label drift",
                        "Mean JS", "Classifier AUC", "Retrain?"]
        st.dataframe(show, hide_index=True, use_container_width=True)
    with c2:
        st.markdown("##### Trigger-policy comparison")
        pol = pd.DataFrame({
            "Policy": ["PSI-only", "Label-only (KS)", "Classifier-only", "ANY (deployed)"],
            "Retrains in": [
                str([int(r["year"]) for _, r in dm.iterrows() if r["trigger_psi_only"]]),
                str([int(r["year"]) for _, r in dm.iterrows() if r["trigger_label_only"]]),
                str([int(r["year"]) for _, r in dm.iterrows() if r["trigger_classifier_only"]]),
                str([int(r["year"]) for _, r in dm.iterrows() if r["trigger_any"]]),
            ]})
        st.dataframe(pol, hide_index=True, use_container_width=True)
        st.caption("Drift fires every year under the deployed policy, so drift-triggered "
                   "~ periodic retraining HERE. The trigger's value is operational: a "
                   "PSI-only policy would skip 2016.")

    if report:
        st.markdown("##### Classifier-identified top drift drivers")
        cols = st.columns(len(report))
        for col, e in zip(cols, report):
            with col:
                st.markdown(f"**{e['year']}**")
                tops = e.get("classifier_top_features", [])[:5]
                st.dataframe(pd.DataFrame(tops).rename(
                    columns={"feature": "feature", "separation_auc": "sep AUC"}),
                    hide_index=True, use_container_width=True)


def page_static_vs_adaptive():
    st.title("Static vs Adaptive - does retraining recover performance?")
    bt = get_csv("backtest_results.csv")
    cmp = get_csv("final_comparison.csv")
    if bt is None:
        st.warning("Run run_phase4.py and run_phase5.py first.")
        return

    tab1, tab2, tab3 = st.tabs(["AUC-ROC", "PR-AUC (threshold-free)", "F1 (threshold note)"])

    def line(df, scol, acol, ytitle, yr):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["year"], y=df[scol], name="Static (trained once)",
                      mode="lines+markers", line=dict(color=ACCENT["static"], width=3)))
        fig.add_trace(go.Scatter(x=df["year"], y=df[acol], name="Adaptive (retrained)",
                      mode="lines+markers", line=dict(color=ACCENT["adaptive"], width=3),
                      fill="tonexty", fillcolor="rgba(124,77,255,0.12)"))
        fig.update_layout(height=430, yaxis_title=ytitle, xaxis_title="Year",
                          legend=dict(orientation="h", y=1.1),
                          xaxis=dict(tickmode="array", tickvals=df["year"].tolist()))
        return fig

    with tab1:
        st.plotly_chart(line(bt, "static_auc", "adaptive_auc", "AUC-ROC", "year"),
                        use_container_width=True)
        st.caption("2015 is the shared anchor (no retraining yet). The gap widens as "
                   "drift accumulates - adaptive recovers ~2015-level discrimination.")
    with tab2:
        if "static_pr_auc" in bt.columns:
            st.plotly_chart(line(bt, "static_pr_auc", "adaptive_pr_auc",
                            "PR-AUC (Average Precision)", "year"), use_container_width=True)
            st.caption("PR-AUC is the proper threshold-free metric for the imbalanced "
                       "default class - adaptive wins every year here.")
    with tab3:
        st.warning("**The F1 subtlety.** At the naive 0.5 cutoff the *static* model shows "
                   "higher F1 - a base-rate artifact (defaults rise to 27%), not skill. "
                   "At a **validation-tuned threshold**, adaptive wins every year.")
        if "static_f1_tuned" in bt.columns:
            fig = go.Figure()
            for col, name, c, dash in [
                ("static_f1_tuned", "Static - F1 @ tuned", ACCENT["static"], None),
                ("adaptive_f1_tuned", "Adaptive - F1 @ tuned", ACCENT["adaptive"], None),
                ("static_f1", "Static - F1 @ 0.5 (artifact)", ACCENT["static"], "dash"),
                ("adaptive_f1", "Adaptive - F1 @ 0.5 (artifact)", ACCENT["adaptive"], "dash")]:
                fig.add_trace(go.Scatter(x=bt["year"], y=bt[col], name=name,
                              mode="lines+markers",
                              line=dict(color=c, width=2.5, dash=dash)))
            fig.update_layout(height=430, yaxis_title="F1", xaxis_title="Year",
                              legend=dict(orientation="h", y=1.12),
                              xaxis=dict(tickmode="array", tickvals=bt["year"].tolist()))
            st.plotly_chart(fig, use_container_width=True)

    if cmp is not None:
        st.markdown("##### Headline comparison (held-out test set, leakage-free)")
        show = cmp.copy()
        for c in show.columns:
            if c != "year":
                show[c] = show[c].round(4)
        st.dataframe(show, hide_index=True, use_container_width=True)


def page_strategies():
    st.title("Retraining strategies - would simpler retraining do as well?")
    strat = get_csv("strategy_comparison.csv")
    if strat is None:
        st.warning("Run run_phase4.py first.")
        return

    st.markdown("Reviewer's question: is the gain from a clever **sliding window**, or "
                "would naive retraining do just as well? We test three retraining "
                "policies against the static baseline on the same test set.")

    names = [("static", ACCENT["static"]), ("naive_recent", ACCENT["orange"]),
             ("sliding", ACCENT["adaptive"]), ("expanding", ACCENT["violet"])]
    fig = go.Figure()
    for name, color in names:
        col = f"{name}_auc"
        if col in strat.columns:
            fig.add_trace(go.Bar(x=strat["year"].astype(str), y=strat[col] * 100,
                          name=name, marker_color=color))
    fig.update_layout(barmode="group", height=460, yaxis_title="AUC-ROC (%)",
                      xaxis_title="Year", yaxis=dict(range=[60, 75]),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    st.success("**Honest finding - recency beats volume.** Naive 'most-recent-year-only' "
               "is strongest, the **expanding** window (all history) is the *weakest* "
               "adaptive option by 2018 - keeping stale 2015-16 data actively hurts once "
               "drift sets in. The deployed **sliding window** is the robust compromise: "
               "it bounds memory without needing the optimal lookback known in advance.")

    show = strat.copy()
    for c in show.columns:
        if c != "year":
            show[c] = show[c].round(4)
    st.dataframe(show, hide_index=True, use_container_width=True)


def page_shap():
    st.title("Explainability stability (SHAP)")
    stab = get_csv("shap_stability_report.csv")
    st.markdown("Does the model's *reasoning* drift, not just its accuracy? We compare "
                "SHAP feature-importance of the static 2015 model across years.")

    if stab is not None:
        c1, c2, c3 = st.columns(3)
        last = stab.iloc[-1]
        c1.metric("Cosine similarity (2018)", f"{last['cosine_similarity']:.3f}",
                  help=">=0.95 = stable overall importance vector.")
        c2.metric("Kendall tau (2018)", f"{last['kendall_tau']:.3f}",
                  help=">=0.8 = stable importance ranking.")
        c3.metric("Jaccard@10 (2018)", f"{last['jaccard_top_n']:.3f}",
                  help="Top-10 membership overlap vs 2015. <1.0 = real churn "
                       "(int_rate climbing in) - caught after fixing a metric bug.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stab["year"], y=stab["cosine_similarity"],
                      name="Cosine similarity", mode="lines+markers",
                      line=dict(color=ACCENT["gold"], width=3)))
        fig.add_trace(go.Scatter(x=stab["year"], y=stab["kendall_tau"],
                      name="Kendall tau", mode="lines+markers",
                      line=dict(color=ACCENT["violet"], width=3)))
        fig.add_trace(go.Scatter(x=stab["year"], y=stab["jaccard_top_n"],
                      name="Jaccard@10", mode="lines+markers",
                      line=dict(color=ACCENT["adaptive"], width=3)))
        fig.update_layout(height=380, yaxis_title="Stability metric", xaxis_title="Year",
                          yaxis=dict(range=[0.5, 1.03]), legend=dict(orientation="h", y=1.12),
                          xaxis=dict(tickmode="array", tickvals=stab["year"].tolist()))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(stab, hide_index=True, use_container_width=True)

    st.markdown("##### Per-year SHAP feature importance")
    years = [2015, 2016, 2017, 2018]
    cols = st.columns(2)
    for i, yr in enumerate(years):
        p = da.plot_path(f"shap_bar_{yr}.png")
        if p:
            cols[i % 2].image(p, caption=f"Mean |SHAP| - {yr}", use_column_width=True)


def page_live():
    st.title("Live drift check & scoring")
    st.markdown("Pick a built-in year (instant demo) **or** upload a window CSV. The app "
                "runs the drift detectors **live** against the 2015 reference and scores "
                "the data with the static vs adaptive models.")

    ref = get_reference()
    if ref is None:
        st.error("2015 reference window not found.")
        return
    ref_cols = da.feature_columns(ref)

    src = st.radio("Data source", ["Use a built-in window", "Upload a CSV"],
                   horizontal=True)
    is_builtin = src == "Use a built-in window"
    current = None
    if is_builtin:
        yrs = [y for y in da.available_window_years() if y != da.REFERENCE_YEAR]
        yr = st.selectbox("Window year (vs 2015 reference)", yrs,
                          index=len(yrs) - 1)
        current = get_window(yr)
    else:
        up = st.file_uploader("Window CSV (same columns as data/windows/window_*.csv)",
                              type="csv")
        if up is not None:
            current = pd.read_csv(up)
        st.caption("Tip: the file should have the 12 numerical features + one-hot "
                   "categoricals (+ optional `target` to get live AUC).")

    adaptive_years = da.available_adaptive_years()
    adaptive_year = st.selectbox("Adaptive model to score with", adaptive_years,
                                 index=len(adaptive_years) - 1)

    if current is None:
        st.info("Choose a built-in window or upload a CSV to begin.")
        return

    st.caption(f"Loaded {len(current):,} rows, {current.shape[1]} columns.")
    if not st.button("Run live drift detection + scoring", type="primary"):
        return

    with st.spinner("Running PSI, KS, Jensen-Shannon, classifier two-sample test..."):
        res = da.run_live_drift(ref, current)
    with st.spinner("Scoring with static and adaptive models..."):
        # Built-in windows: score on the held-out test split (leakage-free, matches
        # the thesis). Uploads are new data, so score in full.
        sc = da.score_models(current, ref, adaptive_year, holdout=is_builtin)

    if res["retrain"]:
        st.error("Drift detected. Retraining recommended.")
    else:
        st.success("No significant drift. The static model is still fit for use.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean PSI", f"{res['mean_psi']:.4f}",
              help=">=0.2 on a feature = critical drift.")
    m2.metric("Mean JS distance", f"{res['mean_js']:.4f}")
    m3.metric("Classifier 2-sample AUC", f"{res['c2st']['drift_auc']:.4f}",
              help="0.5 = indistinguishable from 2015; ->1.0 = strong covariate drift.")
    if res["label"]:
        m4.metric("Default rate", fmt_pct(res["label"]["current_default_rate"]),
                  delta=fmt_pct(res["label"]["current_default_rate"]
                                - res["label"]["reference_default_rate"]),
                  help="vs 2015 reference; KS p<0.05 -> label drift.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Per-feature PSI vs 2015")
        pf = pd.DataFrame({"feature": list(res["psi"]),
                           "PSI": list(res["psi"].values()),
                           "JS": [res["js"].get(f, np.nan) for f in res["psi"]]})
        pf = pf.sort_values("PSI", ascending=False)
        st.dataframe(pf, hide_index=True, use_container_width=True, height=320)
        if res["flagged"]:
            st.caption(f"Critical (PSI>=0.2): **{', '.join(res['flagged'])}**")
    with c2:
        st.markdown("##### Top multivariate drift drivers (classifier)")
        st.dataframe(pd.DataFrame(res["c2st_top"]), hide_index=True,
                     use_container_width=True, height=320)

    st.markdown("### Model scoring on this data")
    st.caption(f"Drift detected on the full window ({len(current):,} rows). "
               f"Scoring evaluated on: **{sc['eval_scope']}** - {sc['n']:,} rows.")
    if sc["missing_cols"]:
        st.warning(f"{len(sc['missing_cols'])} expected column(s) missing - filled with "
                   f"0: {sc['missing_cols'][:8]}{'...' if len(sc['missing_cols'])>8 else ''}")

    cards = st.columns(len(sc["models"]))
    fig = go.Figure()
    palette = [ACCENT["static"], ACCENT["adaptive"]]
    for i, (name, e) in enumerate(sc["models"].items()):
        with cards[i]:
            st.markdown(f"**{name}**")
            if sc["has_labels"]:
                st.metric("AUC-ROC", f"{e['auc']:.4f}")
                st.metric("PR-AUC", f"{e['pr_auc']:.4f}")
                st.metric("F1 @ tuned thr", f"{e['f1_tuned']:.4f}")
            st.metric("Predicted default rate", fmt_pct(e["pred_rate"]),
                      help=f"at tuned threshold {e['threshold']:.3f}")
        fig.add_trace(go.Histogram(x=e["proba"], name=name, opacity=0.6,
                      marker_color=palette[i % len(palette)], nbinsx=40))

    if sc["has_labels"] and len(sc["models"]) == 2:
        vals = list(sc["models"].values())
        gain = vals[1]["auc"] - vals[0]["auc"]
        st.success(f"**Adaptive lift on this data: {gain:+.4f} AUC** "
                   f"({vals[0]['auc']:.4f} -> {vals[1]['auc']:.4f}).")
    fig.update_layout(barmode="overlay", height=360,
                      xaxis_title="Predicted default probability", yaxis_title="Count",
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)


PAGES = {
    "Overview": page_overview,
    "Drift Detection": page_drift,
    "Static vs Adaptive": page_static_vs_adaptive,
    "Retraining Strategies": page_strategies,
    "Explainability (SHAP)": page_shap,
    "Live Drift Check": page_live,
}

st.sidebar.title("Adaptive Credit Risk")
st.sidebar.caption("Concept-drift monitoring & adaptive retraining")
choice = st.sidebar.radio("Navigate", list(PAGES.keys()))
st.sidebar.markdown("---")
st.sidebar.caption("Built on the outputs of run_phase3 -> run_phase5. "
                   "The Live page runs detectors & models in real time.")
PAGES[choice]()
