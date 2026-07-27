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
# INJECT PREMIUM CSS & DYNAMIC THEME
# ===========================================================================
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
try:
    with open(css_path, "r", encoding="utf-8") as f:
        base_css = f.read()
        
    is_dark = st.session_state.get("dark_mode", True)
    
    if is_dark:
        theme_css = """
        :root {
            --text-color: #e8eaed;
            --background-color: #0f1117;
            --secondary-background-color: #1a1d2e;
            --primary-color: #34d399;
        }
        .stSelectbox label, .stNumberInput label, .stRadio label { color: #e8eaed !important; }
        .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input { 
            background-color: rgba(26, 29, 46, 0.7) !important;
            color: #e8eaed !important;
            border-color: rgba(255, 255, 255, 0.2) !important;
        }
        .stSelectbox div[data-baseweb="popover"] {
            background-color: #1a1d2e !important;
        }
        li[role="option"] {
            color: #e8eaed !important;
        }
        """
    else:
        theme_css = """
        :root {
            --text-color: #0f172a;
            --background-color: #f8fafc;
            --secondary-background-color: #ffffff;
            --primary-color: #059669;
            
            --bg-primary: #f8fafc !important;
            --bg-secondary: #ffffff !important;
            --bg-card: rgba(255, 255, 255, 0.9) !important;
            --bg-card-hover: rgba(255, 255, 255, 1) !important;
            --border-card: rgba(0, 0, 0, 0.1) !important;
            --text-primary: #0f172a !important;
            --text-secondary: #334155 !important;
            --text-muted: #64748b !important;
            --accent-green: #059669 !important;
            --accent-green-glow: rgba(5, 150, 105, 0.15) !important;
            --accent-blue: #2563eb !important;
            --accent-purple: #7c3aed !important;
            --accent-amber: #d97706 !important;
            --accent-red: #dc2626 !important;
            --gradient-hero: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 30%, #e0f2fe 70%, #ede9fe 100%) !important;
            --gradient-card: linear-gradient(135deg, rgba(52, 211, 153, 0.15), rgba(96, 165, 250, 0.1)) !important;
            --gradient-accent: linear-gradient(135deg, #10b981, #3b82f6) !important;
        }
        .stApp { background: var(--bg-primary) !important; }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%) !important;
            border-right: 1px solid rgba(0, 0, 0, 0.1) !important;
        }
        .hero-title, .sidebar-logo {
            background: linear-gradient(135deg, #059669 0%, #2563eb 50%, #7c3aed 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
        }
        .prediction-card {
            background: linear-gradient(135deg, rgba(5, 150, 105, 0.1), rgba(15, 23, 42, 0.05)) !important;
            border-color: rgba(5, 150, 105, 0.2) !important;
        }
        .production-card {
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.1), rgba(15, 23, 42, 0.05)) !important;
            border-color: rgba(37, 99, 235, 0.2) !important;
        }
        .stSelectbox label, .stNumberInput label, .stRadio label { color: #0f172a !important; }
        .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input { 
            background-color: #ffffff !important;
            color: #0f172a !important;
            border-color: rgba(0, 0, 0, 0.2) !important;
        }
        .stSelectbox div[data-baseweb="popover"] {
            background-color: #ffffff !important;
        }
        li[role="option"] {
            color: #0f172a !important;
        }
        """
        
    st.markdown(f"<style>{base_css}\n{theme_css}</style>", unsafe_allow_html=True)
except Exception as e:
    st.warning(f"Failed to load styling assets: {e}")

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
