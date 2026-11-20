"""
Model Performance & Error Analysis page.

Everything here is read from models/evaluation.json, which
scripts/train_lightgbm.py writes from the 2018-2020 hold-out predictions.
No hard-coded metrics.
"""

import pandas as pd
import streamlit as st

from src.utils.model_metrics import load_evaluation, headline_metrics
from src.streamlit_app.components.cards import metric_card, section_header
from src.streamlit_app.components.charts import benchmark_r2_bar


def _benchmark_table(bench: dict) -> pd.DataFrame:
    rows = []
    for model, res in bench.items():
        for key, label in [("full", "All test rows"), ("quality_filtered", "Quality-filtered")]:
            m = res[key]
            rows.append({"Model": model, "Test set": label, "Rows": f"{m['n']:,}", "R²": m["r2"],
                         "MAE": m["mae"], "RMSE": m["rmse"], "Median APE (%)": m["median_ape_pct"]})
    return pd.DataFrame(rows)


def render():
    """Render the Model Performance page."""
    ev, hm = load_evaluation(), headline_metrics()
    section_header("📈 Model Performance & Error Analysis",
                   "Out-of-time evaluation: trained on 2001–2017, tested on unseen 2018–2020 data")
    if not ev or not hm:
        st.warning("models/evaluation.json was not found. Run `python scripts/train_lightgbm.py` to generate it.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("R² · quality-filtered test", f"{hm['r2_clean']:.3f}", "🎯")
    with c2:
        metric_card("MAE vs naive baseline", f"−{hm['mae_gain_vs_baseline_pct']:.1f}%", "📉")
    with c3:
        metric_card("R² · all test rows", f"{hm['r2_full']:.3f}", "📊")
    with c4:
        metric_card("Optuna trials", str(ev["optuna"]["trials"]), "🔧")

    section_header("🏁 Benchmark: Before vs After the Quality Filter",
                   f"Production model: {ev['production_model']} · baseline = {ev['baseline_description']}")
    st.plotly_chart(benchmark_r2_bar(ev["benchmark"]), use_container_width=True)
    st.dataframe(_benchmark_table(ev["benchmark"]), use_container_width=True, hide_index=True)

    ea = ev["error_analysis"]
    section_header("🔍 Error Analysis: Why the Raw R² Is Lower")
    st.markdown(f"""
    <div class="info-card">
        <div class="info-text">
            <ul>
                <li><strong>{ea['flagged_rows']} test rows ({ea['flagged_pct_of_test']}%)</strong> report yields above
                    <strong>{ev['quality_rule'].split('drop test rows with yield > ')[-1]}</strong>,
                    which is physically implausible and most likely a data-entry or unit error.</li>
                <li>Those rows alone cause <strong>{ea['flagged_share_of_squared_error_pct']}% of the total squared error</strong>,
                    so every model, including the naive baseline, sits near R² 0.60 on the raw test set.</li>
                <li>The thresholds are fitted on <strong>training years only</strong> (2001–2017) and saved in
                    <code>models/quality_thresholds.json</code>. No test information is used to set them.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Largest errors on the raw test set** (actual vs the crop's highest training yield):")
    worst = pd.DataFrame(ea["worst_errors"]).rename(columns={
        "crop": "Crop", "year": "Year", "actual": "Reported yield (T/Ha)",
        "predicted": "Predicted (T/Ha)", "train_max": "Max in 2001–2017 (T/Ha)"})
    worst["Crop"] = worst["Crop"].str.title()
    st.dataframe(worst, use_container_width=True, hide_index=True)

    section_header("🛡 Validation Design")
    s = ev["split"]
    st.markdown(f"""
    <div class="info-card">
        <div class="info-text">
            <ul>
                <li><strong>Chronological split:</strong> train {s['train_years']} ({s['n_train']:,} rows) →
                    test {s['test_years']} ({s['n_test']:,} rows). No random shuffling across years.</li>
                <li><strong>Hyperparameter tuning:</strong> Optuna on a separate dev split ({s['dev_split']});
                    best dev R² {ev['optuna']['best_dev_r2']}. The test years were scored once, after tuning.</li>
                <li><strong>Leak-free features ({s['n_features']}):</strong> lagged yields use only previous years
                    (expanding/rolling/EWMA, shifted by one year); target encodings are out-of-fold on the training
                    years; KMeans geo-clusters and climate PCA are fitted on training years only.</li>
                <li><strong>Target:</strong> log1p(yield) with a per-crop smearing correction, so back-transformed
                    predictions are not biased low.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"Evaluation generated on {ev['generated']} by scripts/train_lightgbm.py")
