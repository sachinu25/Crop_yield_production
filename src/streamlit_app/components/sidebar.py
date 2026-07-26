import streamlit as st

def render_sidebar():
    """Render the premium sidebar navigation."""
    with st.sidebar:
        st.markdown('''
            <div class="sidebar-logo">Agri AI</div>
            <div class="sidebar-tagline">Intelligent Yield Prediction</div>
        ''', unsafe_allow_html=True)
        
        # Navigation
        app_mode = st.radio(
            "Navigation",
            [
                "🏠 Prediction",
                "📊 Analytics",
                "🌦 Weather",
                "📜 History",
                "ℹ About"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown('<hr class="premium-divider">', unsafe_allow_html=True)
        
        # Theme toggle
        st.toggle("Dark Mode", value=True, key="dark_mode")
        
        st.markdown('<div style="flex-grow: 1;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-version">v2.0.0 Pro</div>', unsafe_allow_html=True)
        
        return app_mode
