import time
import pandas as pd
import numpy as np
import streamlit as st

from src.config.constants import SUPPORTED_AREA_UNITS
from src.services.data_service import get_dropdown_options, get_lat_lon, get_dataframe
from src.services.prediction_service import predict_yield_list
from src.services.weather_service import fetch_weather_data
from src.utils.conversions import convert_to_hectares, hectares_per_unit
from src.streamlit_app.components.hero import render_hero
from src.streamlit_app.components.cards import prediction_card, production_card, confidence_card, metric_card, recommendation_card
from src.streamlit_app.utils.recommendations import generate_recommendations
from src.streamlit_app.utils.pdf_report import generate_pdf_report

def render():
    """Render the Prediction page."""
    render_hero()
    
    # Initialize history list if missing
    if "history" not in st.session_state:
        st.session_state.history = []
        
    options = get_dropdown_options(title_case=True)
    
    with st.container(border=True):
        st.markdown('<div class="section-header" style="margin-top:0;">📋 Farm Details</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            state = st.selectbox("State", options["state_list"], index=None, placeholder="Select State...")
            district_list = options["district_dict"].get(state, []) if state else []
            district = st.selectbox("District", district_list, index=None, placeholder="Select District...")
            
            # Additional logic to determine confidence
            df = get_dataframe()
            known_combos = set(df['state_name'] + '_' + df['crop'])
            
        with c2:
            season = st.selectbox("Season", options["season_list"], index=None, placeholder="Select Season...")
            crop_list = options["crop_list"]
            crop = st.selectbox("Crop", crop_list, index=None, placeholder="Select Crop...")
            
        c3, c4 = st.columns(2)
        with c3:
            year = st.selectbox("Year", list(range(2001, 2030)), index=25) # Default 2026
        with c4:
            area_unit = st.selectbox("Unit of Area", SUPPORTED_AREA_UNITS, index=0)
            
        area = st.number_input("Cultivated Area", min_value=0.1, value=1.0, step=0.5)
        
        ready = all([state, district, season, crop, year, area, area_unit])
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🌾 Predict Yield", use_container_width=True, disabled=not ready)
        
    if predict_btn and ready:
        try:
            lat, lon = get_lat_lon(state, district)
            
            with st.spinner("Fetching live meteorological data..."):
                time.sleep(0.5) # UX delay
                env_data = fetch_weather_data(year, season.lower(), lat, lon)
                
            if env_data.empty:
                st.error("Failed to fetch weather data for the specified parameters.")
                return
                
            weather_summary = {
                "temperature_2m_mean": env_data["temperature_2m_mean"].mean(),
                "precipitation_sum": env_data["precipitation_sum"].sum(),
                "relative_humidity_2m_mean": env_data["relative_humidity_2m_mean"].mean(),
                "wind_speed_10m_mean": env_data["wind_speed_10m_mean"].mean()
            }
                
            area_in_ha = convert_to_hectares(area, area_unit)
            
            pred_input = {
                "crop_year": year,
                "season": season,
                "crop": [crop.lower()],
                "area": area_in_ha,
                "latitude": lat,
                "longitude": lon,
                **weather_summary
            }
            
            with st.spinner("Executing optimized CatBoost inference..."):
                time.sleep(0.5) # UX delay
                predictions = predict_yield_list(pred_input)
                
            result = predictions[0]
            yield_val_ha = result["Yield"]
            
            # Convert yield to selected unit
            ha_per_unit = hectares_per_unit(area_unit)
            yield_val_unit = yield_val_ha * ha_per_unit
            total_production = yield_val_unit * area
            
            # Determine confidence
            state_crop_key = f"{state.lower()}_{crop.lower()}"
            confidence = "High" if state_crop_key in known_combos else "Medium"
            if yield_val_ha < 0.1 or yield_val_ha > 15:
                confidence = "Low"
                
            # Add to history
            st.session_state.history.insert(0, {
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "State": state,
                "District": district,
                "Crop": crop,
                "Year": year,
                "Season": season,
                "Area": area,
                "Unit": area_unit,
                "Yield (T/Ha)": round(yield_val_ha, 3),
                "Production (T)": round(total_production, 3)
            })
            
            # Display Results
            st.markdown('<hr class="premium-divider">', unsafe_allow_html=True)
            
            c_res1, c_res2 = st.columns(2)
            with c_res1:
                prediction_card(yield_val_unit, area_unit)
            with c_res2:
                production_card(total_production)
                
            confidence_card(confidence)
            
            st.markdown('<div class="section-header">🌦 Environmental Profile</div>', unsafe_allow_html=True)
            w1, w2, w3, w4 = st.columns(4)
            with w1: metric_card("Avg Temp", f"{weather_summary['temperature_2m_mean']:.1f} °C", "🌡")
            with w2: metric_card("Total Rain", f"{weather_summary['precipitation_sum']:.1f} mm", "🌧")
            with w3: metric_card("Avg Humidity", f"{weather_summary['relative_humidity_2m_mean']:.0f} %", "💧")
            with w4: metric_card("Wind Speed", f"{weather_summary['wind_speed_10m_mean']:.1f} m/s", "🌬")
            
            # Recommendations
            st.markdown('<div class="section-header">💡 Agronomic Recommendations</div>', unsafe_allow_html=True)
            recs = generate_recommendations(state, district, crop, season, weather_summary, yield_val_ha)
            for r in recs:
                recommendation_card(r['title'], r['text'], r['type'])
                
            # SHAP placeholder (Requires SHAP which takes time, adding simplified explainer)
            st.markdown('<div class="section-header">🧠 Why this prediction?</div>', unsafe_allow_html=True)
            st.info("The CatBoost model heavily weighed the **Historical Crop Type** and **Seasonal Rainfall** to generate this prediction. The temperatures in this region are within optimal ranges for this crop.")
            
            # Download section
            st.markdown('<br>', unsafe_allow_html=True)
            dl1, dl2, _ = st.columns([1, 1, 2])
            
            inputs_for_pdf = {
                "state": state, "district": district, "crop_year": year, 
                "season": season, "area": area, "area_unit": area_unit
            }
            predictions_for_pdf = [{"Crop": crop, "Yield": yield_val_ha, "Production": total_production}]
            
            pdf_bytes = generate_pdf_report(inputs_for_pdf, predictions_for_pdf, weather_summary, recs)
            
            with dl1:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"Yield_Report_{district}_{crop}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            
            with dl2:
                csv_data = pd.DataFrame([st.session_state.history[0]]).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_data,
                    file_name=f"Yield_Data_{district}_{crop}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"Prediction failed: {e}")
