import streamlit as st

def render_hero():
    """Render the premium hero section."""
    html = """
    <div class="hero-container">
        <div class="hero-title">🌾 AI-Powered Crop Yield Prediction System</div>
        <div class="hero-subtitle">
            Predict crop yield using an optimized CatBoost machine learning model 
            with weather insights and intelligent analytics.
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
