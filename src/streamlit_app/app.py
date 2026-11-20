"""
Agri AI — Premium Crop Yield Prediction Platform v2.0

Production-ready Streamlit application with:
  - Light mode default, dark mode toggle
  - 7 pages: Home, Prediction, Analytics, Model Performance, Weather, History, About
  - Optuna-tuned LightGBM model with native SHAP explanations
  - Professional CSS design system
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
from src.streamlit_app.pages import home, prediction, analytics, performance, weather, history, about

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

    is_dark = st.session_state.get("dark_mode", False)

    if is_dark:
        theme_css = """
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --bg-card: #1e293b;
            --bg-card-hover: #334155;
            --border-default: #334155;
            --border-light: #1e293b;
            --border-card: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --text-inverse: #0f172a;
            --accent-green-light: rgba(5, 150, 105, 0.15);
            --accent-blue-light: rgba(37, 99, 235, 0.15);
            --accent-purple-light: rgba(124, 58, 237, 0.15);
            --accent-amber-light: rgba(217, 119, 6, 0.15);
            --accent-red-light: rgba(220, 38, 38, 0.15);
            --accent-green-bg: rgba(5, 150, 105, 0.1);
            --accent-blue-bg: rgba(37, 99, 235, 0.1);
            --accent-purple-bg: rgba(124, 58, 237, 0.1);
            --gradient-hero: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
            --gradient-sidebar: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            --shadow-xs: 0 1px 2px rgba(0, 0, 0, 0.2);
            --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.3);
            --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .stApp { background: var(--bg-primary) !important; }
        section[data-testid="stSidebar"] {
            background: var(--gradient-sidebar) !important;
            border-right: 1px solid var(--border-default) !important;
        }
        .stSelectbox label, .stNumberInput label, .stRadio label, .stTextInput label {
            color: var(--text-primary) !important;
        }
        div[data-baseweb="select"] > div {
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
            border-color: var(--border-default) !important;
        }
        .stNumberInput input {
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
            border-color: var(--border-default) !important;
        }
        div[data-baseweb="popover"] {
            background-color: var(--bg-card) !important;
        }
        li[role="option"] {
            color: var(--text-primary) !important;
        }
        li[role="option"]:hover {
            background-color: var(--bg-tertiary) !important;
        }
        ::-webkit-scrollbar-track { background: var(--bg-primary); }
        ::-webkit-scrollbar-thumb { background: #475569; }
        .stTextInput input {
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
            border-color: var(--border-default) !important;
        }
        /* Plotly dark overrides */
        .js-plotly-plot .plotly .main-svg { background: transparent !important; }
        """
    else:
        # Light mode — CSS defaults are already light, minimal overrides needed
        theme_css = """
        .stSelectbox label, .stNumberInput label, .stRadio label, .stTextInput label {
            color: var(--text-primary) !important;
        }
        div[data-baseweb="select"] > div {
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
            border-color: var(--border-default) !important;
        }
        .stNumberInput input {
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
            border-color: var(--border-default) !important;
        }
        div[data-baseweb="popover"] {
            background-color: var(--bg-card) !important;
        }
        li[role="option"] {
            color: var(--text-primary) !important;
        }
        """

    st.markdown(f"<style>{base_css}\n{theme_css}</style>", unsafe_allow_html=True)
except Exception:
    pass  # Graceful fallback — app works without custom CSS

# ===========================================================================
# ROUTING
# ===========================================================================
app_mode = render_sidebar()

if app_mode == "🏠 Home":
    home.render()
elif app_mode == "🔮 Prediction":
    prediction.render()
elif app_mode == "📊 Analytics":
    analytics.render()
elif app_mode == "📈 Model Performance":
    performance.render()
elif app_mode == "🌦 Weather":
    weather.render()
elif app_mode == "📜 History":
    history.render()
elif app_mode == "ℹ About":
    about.render()
