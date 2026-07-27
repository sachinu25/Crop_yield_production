"""
Premium hero section component.

Two modes:
  - Full hero: Landing/home page with badges and stats
  - Mini hero: Shown on prediction page (compact)
"""

import streamlit as st


def render_hero(mini: bool = False):
    """Render the hero section.

    Args:
        mini: If True, renders a compact version for inner pages.
    """
    if mini:
        html = """
        <div class="hero-container" style="padding: 1.5rem 2rem 1.25rem;">
            <div class="hero-title" style="font-size: 1.4rem;">🌾 Crop Yield Prediction</div>
            <div class="hero-subtitle" style="font-size: 0.88rem;">
                Enter farm details below to generate an AI-powered yield forecast using our production CatBoost model.
            </div>
        </div>
        """
    else:
        html = """
        <div class="hero-container">
            <div class="hero-title">🌾 AI-Powered Crop Yield Prediction</div>
            <div class="hero-subtitle">
                Enterprise-grade agricultural intelligence platform powered by CatBoost ML. 
                Predict crop yields across 33 Indian states with weather-integrated, 
                explainable AI and 60 engineered features.
            </div>
            <div class="hero-badges">
                <span class="hero-badge"><span class="badge-dot"></span> CatBoost AI Model</span>
                <span class="hero-badge"><span class="badge-dot"></span> 60 Engineered Features</span>
                <span class="hero-badge"><span class="badge-dot"></span> Weather Integrated</span>
                <span class="hero-badge"><span class="badge-dot"></span> SHAP Explainability</span>
                <span class="hero-badge"><span class="badge-dot"></span> Production Ready</span>
            </div>
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)


def render_full_hero(total_records: int, states: int, crops: int, r2: float):
    """Render the full home page hero with real stats."""
    html = f"""
    <div class="hero-container">
        <div class="hero-title">🌾 AI-Powered Crop Yield Prediction</div>
        <div class="hero-subtitle">
            Enterprise-grade agricultural intelligence platform powered by CatBoost ML. 
            Predict crop yields across {states} Indian states with weather-integrated, 
            explainable AI and 60 engineered features.
        </div>
        <div class="hero-badges">
            <span class="hero-badge"><span class="badge-dot"></span> CatBoost AI Model</span>
            <span class="hero-badge"><span class="badge-dot"></span> 60 Engineered Features</span>
            <span class="hero-badge"><span class="badge-dot"></span> Weather Integrated</span>
            <span class="hero-badge"><span class="badge-dot"></span> SHAP Explainability</span>
            <span class="hero-badge"><span class="badge-dot"></span> Production Ready</span>
        </div>
        <div class="hero-stats">
            <div class="hero-stat">
                <div class="hero-stat-value">{total_records:,}</div>
                <div class="hero-stat-label">Training Records</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value">{states}</div>
                <div class="hero-stat-label">States Covered</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value">{crops}</div>
                <div class="hero-stat-label">Crop Varieties</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value">{r2:.2%}</div>
                <div class="hero-stat-label">Model R² Score</div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
