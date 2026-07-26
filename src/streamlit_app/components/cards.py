import streamlit as st

def metric_card(title: str, value: str, icon: str):
    """Render a premium glassmorphism metric card."""
    html = f"""
    <div class="metric-card">
        <span class="metric-icon">{icon}</span>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def prediction_card(yield_value: float, unit: str):
    """Render the large hero prediction result card."""
    html = f"""
    <div class="prediction-card">
        <span class="pred-icon">🌾</span>
        <div class="pred-label">Predicted Crop Yield</div>
        <div class="pred-value">{yield_value:,.2f}</div>
        <div class="pred-unit">Tonnes per {unit}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def production_card(production_value: float):
    """Render the estimated total production card."""
    html = f"""
    <div class="production-card">
        <span class="pred-icon">📦</span>
        <div class="pred-label">Estimated Total Production</div>
        <div class="pred-value">{production_value:,.2f}</div>
        <div class="pred-unit">Tonnes</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def confidence_card(confidence: str):
    """Render a confidence badge (High, Medium, Low)."""
    color_class = {
        "High": "confidence-high",
        "Medium": "confidence-medium",
        "Low": "confidence-low"
    }.get(confidence, "confidence-medium")
    
    icon = {
        "High": "🟢",
        "Medium": "🟡",
        "Low": "🔴"
    }.get(confidence, "🟡")
    
    html = f"""
    <div style="text-align: center; margin-top: 1.5rem; margin-bottom: 2rem;">
        <span class="confidence-badge {color_class}">
            {icon} Prediction Confidence: {confidence}
        </span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def recommendation_card(title: str, text: str, type: str = "info"):
    """
    Render a recommendation card.
    type can be 'info', 'warning', 'danger'.
    """
    html = f"""
    <div class="rec-card {type}">
        <div class="rec-title">{title}</div>
        <div class="rec-text">{text}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
