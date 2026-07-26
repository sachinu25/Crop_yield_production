import streamlit as st

def render():
    """Render the About page."""
    st.markdown('<div class="section-header">ℹ Project Overview</div>', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="info-card">
        <div class="info-title">🌾 Agri AI Platform</div>
        <div class="info-text">
            This application is an enterprise-grade crop yield prediction system designed to empower farmers, policymakers, and agricultural analysts. It integrates 20 years of historical agricultural data with live weather APIs to forecast crop productivity across India.
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown('''
        <div class="info-card">
            <div class="info-title">📊 The Dataset</div>
            <div class="info-text">
                <ul>
                    <li><strong>Timeframe:</strong> 2001 - 2020</li>
                    <li><strong>Coverage:</strong> 33 Indian States & Union Territories</li>
                    <li><strong>Scope:</strong> Over 90 distinct crop varieties</li>
                    <li><strong>Scale:</strong> ~350,000 cultivation records</li>
                </ul>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
    with c2:
        st.markdown('''
        <div class="info-card">
            <div class="info-title">🤖 The ML Model</div>
            <div class="info-text">
                <ul>
                    <li><strong>Algorithm:</strong> CatBoost Regressor</li>
                    <li><strong>Performance:</strong> R² = 0.6047, RMSE = 10.01</li>
                    <li><strong>Features:</strong> Native categorical handling with CTR</li>
                    <li><strong>Architecture:</strong> Fault-tolerant with in-memory processing</li>
                </ul>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('<div class="section-header">⚙ Technology Stack</div>', unsafe_allow_html=True)
    st.markdown('''
    <div>
        <span class="tech-badge">Python 3.12</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">CatBoost</span>
        <span class="tech-badge">Plotly</span>
        <span class="tech-badge">SHAP</span>
        <span class="tech-badge">Pandas & NumPy</span>
        <span class="tech-badge">Open-Meteo API</span>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">👨‍💻 Author</div>', unsafe_allow_html=True)
    st.markdown('''
    <div style="display: flex; gap: 1rem; margin-top: 1rem;">
        <a href="https://github.com" target="_blank" style="text-decoration: none;">
            <button style="background: rgba(255,255,255,0.1); color: white; border: 1px solid rgba(255,255,255,0.2); padding: 0.5rem 1.5rem; border-radius: 8px; cursor: pointer; font-weight: 500;">
                GitHub
            </button>
        </a>
        <a href="https://linkedin.com" target="_blank" style="text-decoration: none;">
            <button style="background: rgba(10, 102, 194, 0.2); color: #60a5fa; border: 1px solid rgba(10, 102, 194, 0.5); padding: 0.5rem 1.5rem; border-radius: 8px; cursor: pointer; font-weight: 500;">
                LinkedIn
            </button>
        </a>
    </div>
    ''', unsafe_allow_html=True)
