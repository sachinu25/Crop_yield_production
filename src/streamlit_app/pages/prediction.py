import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

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
            year = st.selectbox("Year", list(range(2021, 2031)), index=5) # Default 2026
        with c4:
            area_unit = st.selectbox("Unit of Area", SUPPORTED_AREA_UNITS, index=0)
            
        area = st.number_input("Cultivated Area", min_value=0.1, value=1.0, step=0.5)
        
        ready = all([state, district, season, crop, year, area, area_unit])
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🌾 Predict Yield", use_container_width=True, disabled=not ready)
        
    if predict_btn and ready:
        try:
            lat, lon = get_lat_lon(state, district)
            
            weather_summary = {}
            with st.spinner("Fetching live meteorological data..."):
                time.sleep(0.5) # UX delay
                try:
                    env_data = fetch_weather_data(year, season.lower(), lat, lon)
                    if not env_data.empty:
                        weather_summary = {
                            "temperature_2m_mean": env_data["temperature_2m_mean"].mean(),
                            "precipitation_sum": env_data["precipitation_sum"].sum(),
                            "relative_humidity_2m_mean": env_data["relative_humidity_2m_mean"].mean(),
                            "wind_speed_10m_mean": env_data["wind_speed_10m_mean"].mean()
                        }
                    else:
                        raise ValueError("Empty weather data")
                except Exception as e:
                    st.warning(f"Weather API failed ({e}). Falling back to historical averages.")
                    df = get_dataframe()
                    hist_weather = df[(df['district_name'] == district.lower()) & (df['season'] == season.lower())]
                    if hist_weather.empty:
                        hist_weather = df[df['district_name'] == district.lower()]
                    if hist_weather.empty:
                        hist_weather = df # global fallback
                    
                    weather_summary = {
                        "temperature_2m_mean": hist_weather["temperature_2m_mean"].mean(),
                        "precipitation_sum": hist_weather["precipitation_sum"].mean(),
                        "relative_humidity_2m_mean": hist_weather["relative_humidity_2m_mean"].mean(),
                        "wind_speed_10m_mean": hist_weather["wind_speed_10m_mean"].mean()
                    }
                
            area_in_ha = convert_to_hectares(area, area_unit)
            
            pred_input = {
                "crop_year": year,
                "season": season,
                "crop": [crop.lower()],
                "area": area_in_ha,
                "latitude": lat,
                "longitude": lon,
                "state_name": state,
                "district_name": district,
                **weather_summary
            }
            
            with st.spinner("Executing optimized CatBoost inference..."):
                time.sleep(0.5) # UX delay
                predictions = predict_yield_list(pred_input)
                
            result = predictions[0]
            yield_val_ha = result["Yield"]
            final_features = result["Features"]
            
            # Weather summary for UI and PDF based on EXACT features used in inference
            weather_summary = {
                "temperature_2m_mean": final_features['temperature_2m_mean'],
                "precipitation_sum": final_features['precipitation_sum'],
                "relative_humidity_2m_mean": final_features['relative_humidity_2m_mean'],
                "wind_speed_10m_mean": final_features['wind_speed_10m_mean']
            }
            
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
                
            # Comparison Chart
            st.markdown('<div class="section-header">📈 Yield Comparison</div>', unsafe_allow_html=True)
            hist_mean = final_features.get('ewma_district_crop', final_features.get('exp_mean_state_crop', yield_val_ha))
            if hist_mean == 0:
                hist_mean = final_features.get('looe_crop', yield_val_ha)
                
            comp_df = pd.DataFrame({
                "Metric": ["Predicted Yield", "Historical Local Average"],
                "Yield (T/Ha)": [yield_val_ha, hist_mean]
            })
            fig_comp = px.bar(comp_df, x="Metric", y="Yield (T/Ha)", text="Yield (T/Ha)", color="Metric",
                              color_discrete_sequence=["#34d399", "#60a5fa"])
            fig_comp.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Feature Importance Explanation
            st.markdown('<div class="section-header">🧠 Why this prediction?</div>', unsafe_allow_html=True)
            st.markdown("This feature importance plot explains the influence of individual features on the final prediction.")
            
            try:
                shap_values = result["ShapValues"]
                feature_names = result["FeatureNames"]
                
                shap_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Impact on Yield": shap_values,
                    "AbsImpact": np.abs(shap_values)
                })
                # Get top 10 most impactful features
                top_shap = shap_df.sort_values("AbsImpact", ascending=False).head(10)
                top_shap = top_shap.sort_values("AbsImpact", ascending=True) # Ascending for horizontal bar
                
                # Color code positive vs negative impact
                top_shap["Impact Type"] = np.where(top_shap["Impact on Yield"] > 0, "Positive", "Negative")
                
                fig_shap = px.bar(
                    top_shap, 
                    x="Impact on Yield", 
                    y="Feature", 
                    orientation="h",
                    color="Impact Type",
                    color_discrete_map={"Positive": "#34d399", "Negative": "#f87171"},
                    title="Top 10 Feature Contributions"
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to generate SHAP plot: {e}")
            
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
