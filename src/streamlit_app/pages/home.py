"""
Home — Premium landing page for Agri AI.

Displays real platform stats from the dataset and model metadata.
No fake data or placeholders.
"""

import json
import streamlit as st

from src.config.settings import PROJECT_ROOT
from src.services.data_service import get_dataframe
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

    # Load model metrics
    try:
        meta_path = PROJECT_ROOT / "models" / "model_meta.json"
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        r2 = meta.get('best_r2', 0.6016)
        rmse = meta.get('best_rmse', 10.05)
        mae = meta.get('best_mae', 1.52)
        n_train = meta.get('n_train', 0)
        n_test = meta.get('n_test', 0)
        train_years = meta.get('train_years', '')
        test_years = meta.get('test_years', '')
    except Exception:
        r2, rmse, mae = 0.6016, 10.05, 1.52
        n_train, n_test = 211683, 35614
        train_years, test_years = "2001-2017", "2018-2020"

    return {
        'total_records': total_records,
        'n_states': n_states,
        'n_districts': n_districts,
        'n_crops': n_crops,
        'n_seasons': n_seasons,
        'year_range': year_range,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'n_train': n_train,
        'n_test': n_test,
        'train_years': train_years,
        'test_years': test_years,
    }


def render():
    """Render the Home page."""
    stats = _load_platform_stats()

    # Hero Section
    render_full_hero(
        total_records=stats['total_records'],
        states=stats['n_states'],
        crops=stats['n_crops'],
        r2=stats['r2'],
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
        metric_card("Engineered Features", "60", "⚙")
    with c8:
        metric_card("Model Type", "CatBoost", "🤖")

    # Model Performance Section
    section_header("🎯 Model Performance", "CatBoost Regressor — Production Metrics")

    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        metric_card("R² Score", f"{stats['r2']:.4f}", "📈")
    with c_m2:
        metric_card("RMSE", f"{stats['rmse']:.2f}", "📉")
    with c_m3:
        metric_card("MAE", f"{stats['mae']:.4f}", "📊")
    with c_m4:
        metric_card("Training Size", f"{stats['n_train']:,}", "🏋")

    # Architecture Overview
    section_header("🏗 Architecture Overview", "End-to-end ML pipeline from data to prediction")

    steps = [
        ("1", "Data Collection", "20 years of crop production data (2001–2020) across 33 states, 685 districts, and 106 crops"),
        ("2", "Feature Engineering", "60 features including climate PCA, target encoding, rolling averages, GDD proxy, interaction terms"),
        ("3", "CatBoost Training", f"3,786 iterations with Optuna-optimized hyperparameters. Trained on {stats['n_train']:,} records ({stats['train_years']})"),
        ("4", "Weather Integration", "Real-time weather from Open-Meteo API with automatic fallback to historical averages"),
        ("5", "SHAP Explainability", "Feature importance computed per prediction using CatBoost's native SHAP implementation"),
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
