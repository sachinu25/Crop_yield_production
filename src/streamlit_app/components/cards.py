"""
Premium card components for the Agri AI dashboard.

All cards render via st.markdown with HTML. No fake data.
"""

import streamlit as st


def metric_card(title: str, value: str, icon: str):
    """Render a clean metric card with icon, value, and label."""
    html = f"""
    <div class="metric-card">
        <span class="metric-icon">{icon}</span>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def prediction_card(yield_value: float, unit: str):
    """Primary prediction result card — green accent."""
    html = f"""
    <div class="prediction-card">
        <span class="pred-icon">🌾</span>
        <div class="pred-label">Predicted Crop Yield</div>
        <div class="pred-value">{yield_value:,.3f}</div>
        <div class="pred-unit">Tonnes per {unit}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def production_card(production_value: float):
    """Total production card — blue accent."""
    html = f"""
    <div class="production-card">
        <span class="pred-icon">📦</span>
        <div class="pred-label">Estimated Total Production</div>
        <div class="pred-value">{production_value:,.3f}</div>
        <div class="pred-unit">Tonnes</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def comparison_metric_card(label: str, value: float, unit: str = "T/Ha",
                           delta: float = None, delta_label: str = ""):
    """Comparison metric card with optional delta indicator."""
    delta_html = ""
    if delta is not None:
        delta_class = "positive" if delta > 0 else "negative" if delta < 0 else "neutral"
        sign = "+" if delta > 0 else ""
        delta_html = f'<div class="comp-delta {delta_class}">{sign}{delta:.2f} {delta_label}</div>'

    html = f"""
    <div class="comparison-card">
        <div class="comp-label">{label}</div>
        <div class="comp-value">{value:.3f} <span style="font-size: 0.75rem; font-weight: 400; color: var(--text-muted);">{unit}</span></div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def confidence_card(confidence: str):
    """Render a confidence badge (Very High, High, Medium, Low)."""
    color_class = {
        "Very High": "confidence-veryhigh",
        "High": "confidence-high",
        "Medium": "confidence-medium",
        "Low": "confidence-low"
    }.get(confidence, "confidence-medium")

    icon = {
        "Very High": "🟢",
        "High": "🟢",
        "Medium": "🟡",
        "Low": "🔴"
    }.get(confidence, "🟡")

    html = f"""
    <div style="text-align: center; margin-top: 1rem; margin-bottom: 1.5rem;">
        <span class="confidence-badge {color_class}">
            {icon} Prediction Confidence: {confidence}
        </span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def recommendation_card(title: str, text: str, type: str = "info", rationale: str = ""):
    """Recommendation card with optional rationale section."""
    rationale_html = ""
    if rationale:
        rationale_html = f'<div class="rec-rationale">💡 {rationale}</div>'

    html = f"""
    <div class="rec-card {type}">
        <div class="rec-title">{title}</div>
        <div class="rec-text">{text}</div>
        {rationale_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def weather_banner(is_live: bool):
    """Show a banner indicating live vs historical weather data source."""
    if is_live:
        html = """
        <div class="weather-banner live">
            🟢 Live weather data from Open-Meteo API
        </div>
        """
    else:
        html = """
        <div class="weather-banner historical">
            ⚠ Live weather unavailable. Historical climate averages are being used.
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)


def executive_metric_table(metrics: list):
    """
    Render a clean metadata table for prediction details.
    metrics: list of (label, value) tuples
    """
    rows = ""
    for label, value in metrics:
        rows += f"<tr><td>{label}</td><td>{value}</td></tr>\n"

    html = f"""
    <div class="exec-dashboard">
        <div class="exec-dashboard-title">Prediction Details</div>
        <table class="pred-meta-table">
            {rows}
        </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def section_header(text: str, subtitle: str = ""):
    """Render a section header with optional subtitle."""
    html = f'<div class="section-header">{text}</div>'
    if subtitle:
        html += f'<div class="section-subheader">{subtitle}</div>'
    st.markdown(html, unsafe_allow_html=True)
