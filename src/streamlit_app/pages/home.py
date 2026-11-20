"""
Home — Premium landing page for Agri AI.

Displays real platform stats from the dataset and model metadata.
No fake data or placeholders.
"""

import streamlit as st

from src.services.data_service import get_dataframe
from src.utils.model_metrics import headline_metrics
from src.streamlit_app.components.hero import render_full_hero
from src.streamlit_app.components.cards import metric_card, section_header


@st.cache_data(ttl=3600)
def _load_platform_stats():
    """Load real stats from dataset and model metadata."""
    df = get_dataframe()
    total_records = len(df)
    n_states = df['state_name'].nunique()
    n_districts = df['district_name'].nunique()
    n_crops = df['crop'].nunique()
    n_seasons = df['season'].nunique()
    year_range = f"{int(df['crop_year'].min())}–{int(df['crop_year'].max())}"

    # Model metrics: single source of truth = models/evaluation.json
    hm = headline_metrics() or {}

    return {
        'total_records': total_records,
        'n_states': n_states,
        'n_districts': n_districts,
        'n_crops': n_crops,
        'n_seasons': n_seasons,
        'year_range': year_range,
        'metrics': hm,
        'n_train': hm.get('n_train', 0),
        'train_years': hm.get('train_years', '2001-2017'),
    }


def render():
    """Render the Home page."""
    stats = _load_platform_stats()

    # Hero Section
    hm = stats['metrics']
    render_full_hero(
        total_records=stats['total_records'],
        states=stats['n_states'],
        crops=stats['n_crops'],
        r2=hm.get('r2_clean'),
    )

    # Quick Stats Row
    section_header("📊 Platform Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Records", f"{stats['total_records']:,}", "📂")
    with c2:
        metric_card("States & UTs", str(stats['n_states']), "🗺")
    with c3:
        metric_card("Districts", str(stats['n_districts']), "📍")
    with c4:
        metric_card("Crop Varieties", str(stats['n_crops']), "🌿")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        metric_card("Seasons", str(stats['n_seasons']), "🗓")
    with c6:
        metric_card("Data Coverage", stats['year_range'], "📅")
    with c7:
        metric_card("Engineered Features", str(hm.get('n_features', 60)), "⚙")
    with c8:
        metric_card("Model Type", "LightGBM", "🤖")

    # Model Performance Section
    section_header("🎯 Model Performance",
                   "Optuna-tuned LightGBM · out-of-time test on 2018–2020 · details on the 📈 Model Performance page")

    if hm:
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        with c_m1:
            metric_card("R² (quality-filtered)", f"{hm['r2_clean']:.3f}", "📈")
        with c_m2:
            metric_card("MAE vs naive baseline", f"−{hm['mae_gain_vs_baseline_pct']:.1f}%", "📉")
        with c_m3:
            metric_card("R² (all test rows)", f"{hm['r2_full']:.3f}", "📊")
        with c_m4:
            metric_card("Training Size", f"{stats['n_train']:,}", "🏋")
    else:
        st.info("Model metrics are unavailable — run scripts/train_lightgbm.py to generate models/evaluation.json.")

    # Architecture Overview
    section_header("🏗 Architecture Overview", "End-to-end ML pipeline from data to prediction")

    steps = [
        ("1", "Data Collection", "20 years of crop production data (2001–2020) across 33 states, 685 districts, and 106 crops"),
        ("2", "Feature Engineering", "60 features including climate PCA, target encoding, rolling averages, GDD proxy, interaction terms"),
        ("3", "LightGBM Training", f"Optuna-tuned on a 2001–2014 / 2015–2017 dev split, refit on {stats['n_train']:,} records ({stats['train_years']}), with a training-derived data-quality filter"),
        ("4", "Weather Integration", "Real-time weather from Open-Meteo API with automatic fallback to historical averages"),
        ("5", "SHAP Explainability", "Per-prediction SHAP waterfall from LightGBM's native TreeSHAP"),
        ("6", "Inference & Dashboard", f"Sub-second prediction with executive dashboard, 10 analytics charts, and PDF reporting"),
    ]

    for num, title, desc in steps:
        st.markdown(f"""
        <div class="arch-step">
            <div class="arch-step-number">{num}</div>
            <div class="arch-step-content">
                <div class="arch-step-title">{title}</div>
                <div class="arch-step-desc">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # CTA
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👉 Navigate to **🔮 Prediction** from the sidebar to start forecasting crop yields.")
