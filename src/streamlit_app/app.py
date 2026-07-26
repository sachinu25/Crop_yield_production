"""
Agri AI — Premium Crop Yield Prediction Platform

Refactored multi-page Streamlit application using modular components, 
premium CSS design system, and the production CatBoost model.
"""

import os
import sys
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import components & pages
from src.streamlit_app.components.sidebar import render_sidebar
from src.streamlit_app.pages import prediction, analytics, weather, history, about

# ===========================================================================
# PAGE CONFIGURATION
# ===========================================================================
st.set_page_config(
    page_title="Agri AI | Crop Yield Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================================================================
# INJECT PREMIUM CSS
# ===========================================================================
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
try:
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception as e:
    st.warning("Failed to load styling assets.")

# ===========================================================================
# ROUTING
# ===========================================================================
app_mode = render_sidebar()

if app_mode == "🏠 Prediction":
    prediction.render()
elif app_mode == "📊 Analytics":
    analytics.render()
elif app_mode == "🌦 Weather":
    weather.render()
elif app_mode == "📜 History":
    history.render()
elif app_mode == "ℹ About":
    about.render()
