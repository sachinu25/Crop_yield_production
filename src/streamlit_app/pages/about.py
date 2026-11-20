"""
About Page — Professional project overview with architecture, ML pipeline, and metrics.

All metrics come from models/evaluation.json (written by scripts/train_lightgbm.py). No hard-coded numbers.
"""

import json
import streamlit as st

from src.config.settings import PROJECT_ROOT
from src.services.data_service import get_dataframe
from src.utils.model_metrics import headline_metrics
from src.streamlit_app.components.cards import metric_card, section_header


def render():
    """Render the About page."""
    section_header("ℹ Project Overview")

    # Load real stats
    df = get_dataframe()
    hm = headline_metrics() or {}
    try:
        with open(PROJECT_ROOT / "models" / "lightgbm_artifacts.json", 'r', encoding='utf-8') as f:
            artifacts = json.load(f)
        lgb_params = artifacts.get('params', {})
        n_rounds = artifacts.get('num_boost_round')
    except Exception:
        lgb_params, n_rounds = {}, None

    # ── Platform Description ──
    st.markdown('''
    <div class="info-card">
        <div class="info-title">🌾 Agri AI — Intelligent Crop Yield Prediction Platform</div>
        <div class="info-text">
            An AI platform for agricultural yield forecasting across India.
            It integrates 20 years of historical crop production data with real-time weather APIs
            to deliver explainable yield predictions. Built with an Optuna-tuned LightGBM model
            and 60 leak-free engineered features including lagged yields, weather anomalies,
            out-of-fold target encoding, and climate PCA.
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
         f"Optuna-tuned LightGBM on a 2001–2014 / 2015–2017 dev split: {n_rounds or '—'} boosting rounds, "
         f"num_leaves={lgb_params.get('num_leaves', '—')}, lr={lgb_params.get('learning_rate', 0):.4f}, "
         f"min_child_samples={lgb_params.get('min_child_samples', '—')}"),
        ("4", "Model Training & Validation",
         f"Trained on {hm.get('n_train', 211683):,} records ({hm.get('train_years', '2001-2017')}). "
         f"Tested out-of-time on {hm.get('n_test', 35614):,} records ({hm.get('test_years', '2018-2020')}) with a "
         f"training-derived quality filter. Target: log1p with per-crop bias correction"),
        ("5", "Weather Integration",
         "Real-time weather from Open-Meteo Historical API (temperature, rainfall, humidity, wind). "
         "Automatic fallback to historical climate averages when live data is unavailable."),
        ("6", "SHAP Explainability",
         "Per-prediction SHAP waterfall using LightGBM's native TreeSHAP. "
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
        fmt = lambda k, d=3: f"{hm[k]:.{d}f}" if k in hm else "—"  # noqa: E731
        st.markdown(f'''
        <div class="info-card">
            <div class="info-title">🤖 Model Performance (out-of-time, 2018–2020)</div>
            <div class="info-text">
                <ul>
                    <li><strong>Algorithm:</strong> LightGBM (Optuna-tuned gradient boosted trees)</li>
                    <li><strong>R², quality-filtered:</strong> {fmt('r2_clean')} · <strong>all rows:</strong> {fmt('r2_full')}</li>
                    <li><strong>MAE, quality-filtered:</strong> {fmt('mae_clean')} T/Ha
                        ({fmt('mae_gain_vs_baseline_pct', 1)}% lower than the naive baseline)</li>
                    <li><strong>Features:</strong> 60 (8 categorical + 52 numerical)</li>
                    <li><strong>Target:</strong> log1p(yield) with per-crop bias correction</li>
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
        <span class="tech-badge">LightGBM</span>
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
        ("LightGBM in production", "Benchmarked against CatBoost and XGBoost out-of-time; LightGBM matches their accuracy with a smaller, faster model and native TreeSHAP, so the app needs no extra SHAP dependency."),
        ("Training-derived quality filter", "Test rows with yields above 2× a crop's 2001–2017 maximum are flagged as data errors; thresholds use training years only, and metrics are reported both before and after filtering."),
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
