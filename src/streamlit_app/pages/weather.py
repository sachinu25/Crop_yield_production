import streamlit as st
from src.services.data_service import get_dropdown_options, get_lat_lon
from src.services.weather_service import fetch_weather_data
from src.streamlit_app.components.cards import metric_card
from src.streamlit_app.components.charts import weather_trend_chart

def render():
    """Render the Weather insights page."""
    st.markdown('<div class="section-header">🌦 Weather Insights</div>', unsafe_allow_html=True)
    st.markdown("Fetch historical API weather data for any district in India to analyze seasonal micro-climates.")
    
    options = get_dropdown_options(title_case=True)
    
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            state = st.selectbox("State", options["state_list"], key="weather_state")
            
        with c2:
            district_list = options["district_dict"].get(state, []) if state else []
            district = st.selectbox("District", district_list, key="weather_dist")
            
        with c3:
            year = st.selectbox("Year", list(range(2001, 2021)), index=19, key="weather_year")
            
        with c4:
            season = st.selectbox("Season", options["season_list"], key="weather_season")
            
        fetch = st.button("Fetch Weather Data", use_container_width=True)
        
    if fetch and state and district and year and season:
        try:
            lat, lon = get_lat_lon(state, district)
            
            with st.spinner("Fetching live weather data from Open-Meteo..."):
                env_data = fetch_weather_data(year, season.lower(), lat, lon)
                
            if env_data.empty:
                st.error("No weather data found for this period.")
                return
                
            # Metric Cards
            st.markdown('<div class="section-header">Seasonal Averages</div>', unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)
            
            with k1:
                metric_card("Avg Temp", f"{env_data['temperature_2m_mean'].mean():.1f} °C", "🌡")
            with k2:
                metric_card("Avg Humidity", f"{env_data['relative_humidity_2m_mean'].mean():.1f} %", "💧")
            with k3:
                metric_card("Total Rainfall", f"{env_data['precipitation_sum'].sum():.1f} mm", "🌧")
            with k4:
                metric_card("Wind Speed", f"{env_data['wind_speed_10m_mean'].mean():.1f} m/s", "🌬")
                
            # Charts
            st.markdown('<div class="section-header">Daily Parameter Trends</div>', unsafe_allow_html=True)
            fig = weather_trend_chart(env_data)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📄 View Raw Data"):
                st.dataframe(env_data, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error fetching data: {e}")
