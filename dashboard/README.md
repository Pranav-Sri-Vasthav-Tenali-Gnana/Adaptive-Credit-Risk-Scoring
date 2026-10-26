# Results Dashboard

Interactive Streamlit UI for the Adaptive Credit Risk Scoring project. It presents
every result and runs drift detection and model scoring live.

## Launch

From the project root on Windows, double-click `run_dashboard.bat`, or run:

```bat
venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

Then open http://localhost:8501. Stop it with Ctrl+C in the terminal.

## Pages

Overview gives the problem statement, headline numbers (static AUC drop, adaptive
AUC and PR-AUC gains), the dataset, and the full text report.

Drift Detection plots PSI, KS, Jensen-Shannon, and the classifier two-sample test
over time, with the trigger-policy comparison and the features the classifier uses
to tell the years apart.

Static vs Adaptive shows AUC, PR-AUC, and F1 over time, with the F1 threshold
artifact explained inline.

Retraining Strategies compares static against naive-recent, sliding, and expanding
retraining, which is where the recency-beats-volume finding shows up.

Explainability (SHAP) shows the importance-stability metrics (cosine, Kendall tau,
and the corrected Jaccard) plus the per-year importance bars.

Live Drift Check lets you pick a built-in window or upload a CSV, then runs the
detectors and scores the static and adaptive models in real time.

## Live page notes

A built-in window is the instant demo path; it reads `data/windows/window_{year}.csv`
and scores on the same held-out test split as the thesis, so the numbers match. An
uploaded CSV needs the same schema as the window files (12 numerical features plus
the one-hot categoricals, with an optional `target` column to get live AUC and
PR-AUC), and is scored in full since it's new data. Missing one-hot columns are
filled with zero. The classifier two-sample test fits a model on the fly, so a live
run takes roughly 10 to 40 seconds depending on row count.

## Data source

The dashboard reads the artifacts produced by `run_phase3.py` through `run_phase5.py`
(the CSV, JSON, and text files under `outputs/reports/`, the charts under
`outputs/plots/`) and the saved models in `outputs/models/`. Re-run those phases to
refresh it.
