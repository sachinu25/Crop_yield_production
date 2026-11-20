"""
Weather Page — Historical weather explorer with graceful error handling.

Fetches weather data from Open-Meteo API. Shows historical averages as fallback.
Never exposes API URLs, Python errors, or exception traces.
"""

import streamlit as st
import pandas as pd

from src.services.data_service import get_dropdown_options, get_lat_lon, get_dataframe
from src.services.weather_service import fetch_weather_data
from src.streamlit_app.components.cards import metric_card, section_header, weather_banner
from src.streamlit_app.components.charts import weather_trend_chart, district_climate_yield_chart


def _render_district_history(state: str, district: str, crop: str | None) -> None:
    """2001-2020 weather anomalies and (optionally) crop yield for the selected district."""
    df = get_dataframe()
    hist = df[(df["state_name"] == state.lower()) & (df["district_name"] == district.lower())]
    if hist.empty:
        return
    yearly = hist.groupby("crop_year").agg(
        temperature=("temperature_2m_mean", "mean"),
        rainfall=("precipitation_sum", "mean"),
    ).reset_index()
    yearly["temp_anomaly"] = yearly["temperature"] - yearly["temperature"].mean()
    yearly["rain_anomaly_pct"] = 100 * (yearly["rainfall"] / yearly["rainfall"].mean() - 1)
    if crop:
        cy = hist[hist["crop"].str.lower() == crop.lower()].groupby("crop_year")["crop_yield"].mean()
        yearly["crop_yield"] = yearly["crop_year"].map(cy)

    section_header("📉 Weather Anomalies vs Yield (2001–2020)",
                   f"{district.title()}, {state.title()}"
                   + (f" · {crop.title()} yield on the right axis" if crop else "")
                   + " · anomalies are relative to the district's 20-year mean")
    st.plotly_chart(district_climate_yield_chart(yearly, crop), use_container_width=True)


def render():
    """Render the Weather insights page."""
    section_header("🌦 Weather Intelligence",
                   "Explore historical weather data for any district across India")

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

        crop = st.selectbox("Crop (optional — overlays its yield on the trend chart)",
                            options["crop_list"], index=None, placeholder="Any crop", key="weather_crop")
        fetch = st.button("🌍 Fetch Weather Data", use_container_width=True)

    if fetch and state and district and year and season:
        try:
            lat, lon = get_lat_lon(state, district)
        except Exception:
            st.error("Could not determine coordinates for this location. Please select a different district.")
            return

        env_data = None
        is_live = False

        with st.spinner("Retrieving weather data..."):
            try:
                env_data = fetch_weather_data(year, season.lower(), lat, lon)
                if env_data is not None and not env_data.empty:
                    is_live = True
            except Exception:
                pass  # Fall through to historical fallback

        if not is_live or env_data is None or env_data.empty:
            # Historical fallback
            df = get_dataframe()
            hist = df[
                (df['district_name'] == district.lower()) &
                (df['season'] == season.lower())
            ]
            if hist.empty:
                hist = df[df['district_name'] == district.lower()]
            if hist.empty:
                hist = df[df['state_name'] == state.lower()]

            if hist.empty:
                st.warning("No weather data available for this location and season.")
                return

            weather_banner(is_live=False)

            section_header("📊 Historical Climate Averages",
                           f"Based on {len(hist):,} historical records for {district.title()}")

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                metric_card("Avg Temperature", f"{hist['temperature_2m_mean'].mean():.1f} °C", "🌡")
            with k2:
                metric_card("Avg Humidity", f"{hist['relative_humidity_2m_mean'].mean():.1f}%", "💧")
            with k3:
                metric_card("Avg Rainfall", f"{hist['precipitation_sum'].mean():.1f} mm", "🌧")
            with k4:
                metric_card("Avg Wind Speed", f"{hist['wind_speed_10m_mean'].mean():.1f} m/s", "🌬")

            # Year-wise breakdown if available
            yearly = hist.groupby('crop_year').agg({
                'temperature_2m_mean': 'mean',
                'precipitation_sum': 'mean',
                'relative_humidity_2m_mean': 'mean',
                'wind_speed_10m_mean': 'mean',
            }).reset_index()

            if len(yearly) > 1:
                section_header("📈 Year-wise Climate Trends")
                yearly_display = yearly.rename(columns={
                    'crop_year': 'Year',
                    'temperature_2m_mean': 'Temp (°C)',
                    'precipitation_sum': 'Rainfall (mm)',
                    'relative_humidity_2m_mean': 'Humidity (%)',
                    'wind_speed_10m_mean': 'Wind (m/s)',
                })
                st.dataframe(yearly_display, use_container_width=True, hide_index=True)

            _render_district_history(state, district, crop)
            return

        # ── Live Data Display ──
        weather_banner(is_live=True)

        section_header("📊 Seasonal Averages",
                       f"{season.title()} {year} — {district.title()}, {state.title()}")

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            metric_card("Avg Temperature", f"{env_data['temperature_2m_mean'].mean():.1f} °C", "🌡")
        with k2:
            metric_card("Avg Humidity", f"{env_data['relative_humidity_2m_mean'].mean():.1f}%", "💧")
        with k3:
            metric_card("Total Rainfall", f"{env_data['precipitation_sum'].sum():.1f} mm", "🌧")
        with k4:
            metric_card("Avg Wind Speed", f"{env_data['wind_speed_10m_mean'].mean():.1f} m/s", "🌬")

        # Daily trend chart
        section_header("📈 Daily Weather Parameters")
        fig = weather_trend_chart(env_data)
        st.plotly_chart(fig, use_container_width=True)

        # Raw data
        with st.expander("📋 View Raw Data"):
            display_df = env_data.copy()
            display_df.columns = ['Lat', 'Lon', 'Date', 'Rainfall (mm)', 'Humidity (%)', 'Temp (°C)', 'Wind (m/s)']
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        _render_district_history(state, district, crop)
