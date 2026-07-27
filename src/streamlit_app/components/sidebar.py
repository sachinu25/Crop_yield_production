"""
Sidebar navigation component.

Clean professional sidebar with navigation, theme toggle, and version badge.
"""

import streamlit as st


def render_sidebar():
    """Render the sidebar and return the selected page name."""
    with st.sidebar:
        st.markdown('''
            <div class="sidebar-logo">Agri AI</div>
            <div class="sidebar-tagline">Intelligent Crop Yield Prediction</div>
        ''', unsafe_allow_html=True)

        # Navigation
        app_mode = st.radio(
            "Navigation",
            [
                "🏠 Home",
                "🔮 Prediction",
                "📊 Analytics",
                "🌦 Weather",
                "📜 History",
                "ℹ About"
            ],
            label_visibility="collapsed"
        )

        st.markdown('<hr class="premium-divider">', unsafe_allow_html=True)

        # Theme toggle — default to Light Mode (False = light)
        st.toggle("Dark Mode", value=False, key="dark_mode")

        st.markdown('<div style="flex-grow: 1;"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-version">v2.0.0 · CatBoost</div>',
            unsafe_allow_html=True
        )

        return app_mode
