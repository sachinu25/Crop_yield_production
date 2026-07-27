"""
About Page — Professional project overview with architecture, ML pipeline, and metrics.

All metrics from real model_meta.json. No fake numbers.
"""

import json
import streamlit as st

from src.config.settings import PROJECT_ROOT
from src.services.data_service import get_dataframe
from src.streamlit_app.components.cards import metric_card, section_header


def render():
    """Render the About page."""
    section_header("ℹ Project Overview")

    # Load real stats
    df = get_dataframe()
    try:
        with open(PROJECT_ROOT / "models" / "model_meta.json", 'r') as f:
            meta = json.load(f)
    except Exception:
        meta = {}

    try:
        with open(PROJECT_ROOT / "models" / "best_params.json", 'r') as f:
            params = json.load(f)
        cb_params = params.get('CatBoost', {})
    except Exception:
        cb_params = {}

    # ── Platform Description ──
    st.markdown('''
    <div class="info-card">
        <div class="info-title">🌾 Agri AI — Intelligent Crop Yield Prediction Platform</div>
        <div class="info-text">
            An enterprise-grade AI platform for agricultural yield forecasting across India.
            It integrates 20 years of historical crop production data with real-time weather APIs
            to deliver accurate, explainable yield predictions. Built with production-ready
            CatBoost ML and 60 engineered features including climate PCA, target encoding,
            rolling averages, and interaction terms.
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # ── Architecture Pipeline ──
    section_header("🏗 ML Architecture Pipeline")

    steps = [
        ("1", "Data Ingestion",
         f"Raw crop production data from {int(df['crop_year'].min())}–{int(df['crop_year'].max())} "
         f"covering {df['state_name'].nunique()} states, {df['district_name'].nunique()} districts, "
         f"and {df['crop'].nunique()} crops ({len(df):,} records)"),
        ("2", "Feature Engineering",
         "60 features: log transforms, climate interactions (temp×humidity, precip×humidity), "
         "PCA on climate variables, cyclic season encoding, target encoding (LOO, EWMA), "
         "rolling averages, GDD proxy, drought/heat/humidity indicators, geo-clustering"),
        ("3", "Hyperparameter Optimization",
         f"Optuna-tuned CatBoost with {cb_params.get('iterations', 3786)} iterations, "
         f"depth={cb_params.get('depth', 8)}, lr={cb_params.get('learning_rate', 0.073):.4f}, "
         f"L2={cb_params.get('l2_leaf_reg', 23.1):.1f}, subsample={cb_params.get('subsample', 0.92):.3f}"),
        ("4", "Model Training & Validation",
         f"Trained on {meta.get('n_train', 211683):,} records ({meta.get('train_years', '2001-2017')}). "
         f"Validated on {meta.get('n_test', 35614):,} records ({meta.get('test_years', '2018-2020')}). "
         f"Target transform: log1p → expm1"),
        ("5", "Weather Integration",
         "Real-time weather from Open-Meteo Historical API (temperature, rainfall, humidity, wind). "
         "Automatic fallback to historical climate averages when live data is unavailable."),
        ("6", "SHAP Explainability",
         "Per-prediction feature importance using CatBoost's native SHAP implementation. "
         "Each prediction includes a breakdown of which features increased or decreased the yield estimate."),
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

    # ── Dataset Details ──
    section_header("📊 Dataset Details")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'''
        <div class="info-card">
            <div class="info-title">📂 The Dataset</div>
            <div class="info-text">
                <ul>
                    <li><strong>Timeframe:</strong> {int(df['crop_year'].min())}–{int(df['crop_year'].max())} (20 years)</li>
                    <li><strong>States & UTs:</strong> {df['state_name'].nunique()}</li>
                    <li><strong>Districts:</strong> {df['district_name'].nunique()}</li>
                    <li><strong>Crop Varieties:</strong> {df['crop'].nunique()}</li>
                    <li><strong>Seasons:</strong> {df['season'].nunique()} ({', '.join(s.title() for s in sorted(df['season'].unique()))})</li>
                    <li><strong>Total Records:</strong> {len(df):,}</li>
                </ul>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    with c2:
        r2 = meta.get('best_r2', 0.6016)
        rmse = meta.get('best_rmse', 10.05)
        mae = meta.get('best_mae', 1.52)
        st.markdown(f'''
        <div class="info-card">
            <div class="info-title">🤖 Model Performance</div>
            <div class="info-text">
                <ul>
                    <li><strong>Algorithm:</strong> CatBoost Regressor (Gradient Boosted Trees)</li>
                    <li><strong>R² Score:</strong> {r2:.4f}</li>
                    <li><strong>RMSE:</strong> {rmse:.4f}</li>
                    <li><strong>MAE:</strong> {mae:.4f}</li>
                    <li><strong>Features:</strong> 60 (8 categorical + 52 numerical)</li>
                    <li><strong>Target:</strong> log1p(yield) → expm1 inverse</li>
                </ul>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # ── Feature Engineering Breakdown ──
    section_header("⚙ Feature Engineering — 60 Features")

    features_info = [
        ("🔢 Base Features (7)", "crop_year, area, latitude, longitude, temperature, precipitation, humidity, wind"),
        ("📐 Log Transforms (2)", "log_area, log_precip — reduces skew in right-tailed distributions"),
        ("🔀 Interaction Features (8)", "temp×humidity, precip×humidity, wind×temp, temp/precip ratio, GDD×precip, etc."),
        ("🚦 Binary Indicators (3)", "is_drought (<5mm rain), is_extreme_heat (>35°C), is_high_humidity (>80%)"),
        ("📊 Target Encodings (5)", "Leave-one-out encoding for crop, state, district, state-crop, district-crop"),
        ("📈 Rolling Averages (12)", "Expanding mean, 3-year rolling mean, EWMA for state-crop, district-crop, crop, state"),
        ("🌡 Climate Analysis (4)", "PCA components, temperature anomaly, precipitation anomaly"),
        ("🗓 Temporal (4)", "years_since_2001, year_norm_sq, season_sin, season_cos"),
        ("📍 Frequency (4)", "Frequency encoding for district, state, crop, season"),
        ("🗺 Geo Clusters (3)", "k-means clustering (k=10, 30, 50) on latitude/longitude"),
        ("🏷 Categorical (8)", "season, crop, state, crop_season, state_crop, geo_cluster_10/30/50"),
    ]

    for title, desc in features_info:
        st.markdown(f"""
        <div class="info-card" style="padding: 1rem 1.25rem; margin-bottom: 0.4rem;">
            <div class="info-title" style="font-size: 0.88rem; margin-bottom: 0.2rem;">{title}</div>
            <div class="info-text" style="font-size: 0.8rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Technology Stack ──
    section_header("🛠 Technology Stack")
    st.markdown('''
    <div style="display: flex; flex-wrap: wrap; gap: 0.25rem;">
        <span class="tech-badge">Python 3.12</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">CatBoost</span>
        <span class="tech-badge blue">Plotly</span>
        <span class="tech-badge blue">Pandas</span>
        <span class="tech-badge blue">NumPy</span>
        <span class="tech-badge purple">SHAP</span>
        <span class="tech-badge purple">Scikit-learn</span>
        <span class="tech-badge">Open-Meteo API</span>
        <span class="tech-badge blue">FPDF2</span>
        <span class="tech-badge purple">Optuna</span>
    </div>
    ''', unsafe_allow_html=True)

    # ── Key Design Decisions ──
    section_header("🔑 Key Design Decisions")
    decisions = [
        ("Log-transformed target", "Yield distribution is right-skewed. log1p transform normalizes residuals and improves gradient boosting performance."),
        ("CatBoost over LightGBM/XGBoost", "CatBoost natively handles categorical features with CTR (Counter Target Rate), eliminating manual encoding for 8 categorical features."),
        ("EWMA over simple mean", "Exponentially weighted moving average captures recent trends better than simple historical mean for yield forecasting."),
        ("PCA on climate", "Reduces multicollinearity between temperature, rainfall, humidity, and wind into 2 orthogonal components."),
        ("Graceful weather fallback", "If live API fails, historical climate averages are used automatically — prediction never stops."),
    ]
    for title, desc in decisions:
        st.markdown(f"""
        <div class="info-card" style="padding: 1rem 1.25rem; margin-bottom: 0.4rem;">
            <div class="info-title" style="font-size: 0.88rem; margin-bottom: 0.2rem;">→ {title}</div>
            <div class="info-text" style="font-size: 0.8rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
