"""
Prediction Page — Executive Dashboard.

After prediction, displays:
  - 15+ real metrics from CatBoost output and dataset
  - 10 professional charts (all from real data)
  - Structured recommendations
  - India map with state highlight
  - PDF and CSV export

NO fake data. NO random values. NO placeholder charts.
"""

import time
import uuid
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st

from src.config.constants import SUPPORTED_AREA_UNITS
from src.services.data_service import get_dropdown_options, get_lat_lon, get_dataframe
from src.services.prediction_service import predict_yield_list
from src.services.weather_service import fetch_weather_data
from src.utils.conversions import convert_to_hectares, hectares_per_unit
from src.streamlit_app.components.hero import render_hero
from src.streamlit_app.components.cards import (
    prediction_card, production_card, confidence_card,
    comparison_metric_card, metric_card, recommendation_card,
    weather_banner, executive_metric_table, section_header,
)
from src.streamlit_app.components.charts import (
    yield_comparison_grouped_bar, yield_trend_line, district_ranking_bar,
    crop_comparison_bar, season_donut, feature_importance_bar,
    confidence_gauge, production_histogram, yield_heatmap, india_map,
)
from src.streamlit_app.utils.recommendations import generate_recommendations
from src.streamlit_app.utils.pdf_report import generate_pdf_report


def _compute_confidence(state_crop_key: str, known_combos: set,
                        yield_val: float, hist_avg: float) -> tuple:
    """Compute confidence level and numeric score from real data."""
    score = 50  # Base
    label = "Medium"

    # Known crop-state combination
    if state_crop_key in known_combos:
        score += 25
    else:
        score -= 15

    # Yield in reasonable range
    if 0.1 <= yield_val <= 50:
        score += 10
    elif yield_val < 0.1 or yield_val > 100:
        score -= 20

    # Historical data available
    if hist_avg and hist_avg > 0:
        score += 10
        # Prediction within reasonable range of historical
        if hist_avg > 0:
            ratio = yield_val / hist_avg
            if 0.3 <= ratio <= 3.0:
                score += 5

    score = max(5, min(95, score))

    if score >= 80:
        label = "Very High"
    elif score >= 60:
        label = "High"
    elif score >= 35:
        label = "Medium"
    else:
        label = "Low"

    return label, score


def render():
    """Render the Prediction page."""
    render_hero(mini=True)

    # Initialize history
    if "history" not in st.session_state:
        st.session_state.history = []

    options = get_dropdown_options(title_case=True)
    df = get_dataframe()
    known_combos = set(df['state_name'] + '_' + df['crop'].str.lower())

    # ── Input Form ──
    with st.container(border=True):
        section_header("📋 Farm Details")

        c1, c2 = st.columns(2)
        with c1:
            state = st.selectbox("State", options["state_list"], index=None, placeholder="Select State...")
            district_list = options["district_dict"].get(state, []) if state else []
            district = st.selectbox("District", district_list, index=None, placeholder="Select District...")
        with c2:
            season = st.selectbox("Season", options["season_list"], index=None, placeholder="Select Season...")
            crop = st.selectbox("Crop", options["crop_list"], index=None, placeholder="Select Crop...")

        c3, c4 = st.columns(2)
        with c3:
            year = st.selectbox("Year", list(range(2021, 2031)), index=5)
        with c4:
            area_unit = st.selectbox("Unit of Area", SUPPORTED_AREA_UNITS, index=0)

        area = st.number_input("Cultivated Area", min_value=0.1, value=1.0, step=0.5)

        ready = all([state, district, season, crop, year, area, area_unit])
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🌾 Predict Yield", use_container_width=True, disabled=not ready)

    # ── Prediction Logic ──
    if predict_btn and ready:
        try:
            lat, lon = get_lat_lon(state, district)

            # Weather fetch with graceful fallback
            weather_summary = {}
            is_live_weather = False
            with st.spinner("Fetching meteorological data..."):
                try:
                    env_data = fetch_weather_data(year, season.lower(), lat, lon)
                    if not env_data.empty:
                        weather_summary = {
                            "temperature_2m_mean": env_data["temperature_2m_mean"].mean(),
                            "precipitation_sum": env_data["precipitation_sum"].sum(),
                            "relative_humidity_2m_mean": env_data["relative_humidity_2m_mean"].mean(),
                            "wind_speed_10m_mean": env_data["wind_speed_10m_mean"].mean(),
                        }
                        is_live_weather = True
                    else:
                        raise ValueError("Empty weather response")
                except Exception:
                    # Graceful fallback to historical averages
                    hist_w = df[
                        (df['district_name'] == district.lower()) &
                        (df['season'] == season.lower())
                    ]
                    if hist_w.empty:
                        hist_w = df[df['district_name'] == district.lower()]
                    if hist_w.empty:
                        hist_w = df[df['state_name'] == state.lower()]
                    if hist_w.empty:
                        hist_w = df

                    weather_summary = {
                        "temperature_2m_mean": float(hist_w["temperature_2m_mean"].mean()),
                        "precipitation_sum": float(hist_w["precipitation_sum"].mean()),
                        "relative_humidity_2m_mean": float(hist_w["relative_humidity_2m_mean"].mean()),
                        "wind_speed_10m_mean": float(hist_w["wind_speed_10m_mean"].mean()),
                    }

            # Convert area
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
                **weather_summary,
            }

            # Run inference with timing
            with st.spinner("Running CatBoost inference..."):
                t_start = time.time()
                predictions = predict_yield_list(pred_input)
                inference_ms = (time.time() - t_start) * 1000

            result = predictions[0]
            yield_val_ha = result["Yield"]
            final_features = result["Features"]
            shap_values = result["ShapValues"]
            feature_names = result["FeatureNames"]

            # Use exact features from inference for weather display
            weather_summary = {
                "temperature_2m_mean": final_features['temperature_2m_mean'],
                "precipitation_sum": final_features['precipitation_sum'],
                "relative_humidity_2m_mean": final_features['relative_humidity_2m_mean'],
                "wind_speed_10m_mean": final_features['wind_speed_10m_mean'],
            }

            # Convert yield to selected unit
            ha_per_unit = hectares_per_unit(area_unit)
            yield_val_unit = yield_val_ha * ha_per_unit
            total_production = yield_val_unit * area

            # ── Compute Real Metrics (clean inf values) ──
            df_clean = df.copy()
            df_clean['crop_yield'] = df_clean['crop_yield'].replace([np.inf, -np.inf], np.nan)
            df_clean['crop_lower'] = df_clean['crop'].str.lower()

            # Historical district average
            district_hist = df_clean[
                (df_clean['district_name'] == district.lower()) &
                (df_clean['crop_lower'] == crop.lower())
            ]
            district_avg = float(district_hist['crop_yield'].mean()) if not district_hist.empty and district_hist['crop_yield'].notna().any() else 0
            if np.isnan(district_avg) or np.isinf(district_avg):
                district_avg = 0

            # State average
            state_hist = df_clean[
                (df_clean['state_name'] == state.lower()) &
                (df_clean['crop_lower'] == crop.lower())
            ]
            state_avg = float(state_hist['crop_yield'].mean()) if not state_hist.empty and state_hist['crop_yield'].notna().any() else 0
            if np.isnan(state_avg) or np.isinf(state_avg):
                state_avg = 0

            # National average for this crop
            crop_national = df_clean[df_clean['crop_lower'] == crop.lower()]
            national_avg = float(crop_national['crop_yield'].mean()) if not crop_national.empty and crop_national['crop_yield'].notna().any() else 0
            if np.isnan(national_avg) or np.isinf(national_avg):
                national_avg = 0

            # Fallback: use lookup-based averages if dataset gives zero
            if district_avg == 0:
                district_avg = float(final_features.get('ewma_district_crop', 0))
            if state_avg == 0:
                state_avg = float(final_features.get('ewma_state_name', 0))
            if national_avg == 0:
                national_avg = float(final_features.get('ewma_crop', 0))

            # Difference and improvement
            diff_vs_hist = yield_val_ha - district_avg if district_avg > 0 else 0
            improvement_pct = ((yield_val_ha - district_avg) / district_avg * 100) if district_avg > 0 else 0

            # Crop rank in state
            state_crop_yields = df_clean[df_clean['state_name'] == state.lower()].groupby('crop_lower')['crop_yield'].mean().dropna()
            state_crop_yields = state_crop_yields.replace([np.inf, -np.inf], np.nan).dropna()
            state_crop_yields = state_crop_yields.sort_values(ascending=False)
            crop_rank = (
                list(state_crop_yields.index).index(crop.lower()) + 1
                if crop.lower() in state_crop_yields.index else 0
            )
            total_crops_in_state = len(state_crop_yields)

            # Confidence
            state_crop_key = f"{state.lower()}_{crop.lower()}"
            confidence_label, confidence_score = _compute_confidence(
                state_crop_key, known_combos, yield_val_ha, district_avg
            )

            # Prediction metadata
            prediction_id = str(uuid.uuid4())[:12].upper()
            prediction_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            weather_source = "Live API (Open-Meteo)" if is_live_weather else "Historical Average"

            # Add to history
            st.session_state.history.insert(0, {
                "Timestamp": prediction_ts,
                "ID": prediction_id,
                "State": state,
                "District": district,
                "Crop": crop,
                "Year": year,
                "Season": season,
                "Area": area,
                "Unit": area_unit,
                "Yield (T/Ha)": round(yield_val_ha, 3),
                "Production (T)": round(total_production, 3),
                "Confidence": confidence_label,
            })

            # ══════════════════════════════════════════════════
            # EXECUTIVE DASHBOARD
            # ══════════════════════════════════════════════════

            st.markdown('<hr class="premium-divider">', unsafe_allow_html=True)

            # Weather source banner
            weather_banner(is_live_weather)

            # Primary Result Cards
            c_res1, c_res2 = st.columns(2)
            with c_res1:
                prediction_card(yield_val_unit, area_unit)
            with c_res2:
                production_card(total_production)

            confidence_card(confidence_label)

            # Comparison Metrics Row
            section_header("📊 Yield Comparison Metrics")
            cm1, cm2, cm3, cm4 = st.columns(4)
            with cm1:
                comparison_metric_card(
                    "District Historical Avg", district_avg,
                    delta=diff_vs_hist, delta_label="T/Ha"
                )
            with cm2:
                comparison_metric_card("State Historical Avg", state_avg)
            with cm3:
                comparison_metric_card("National Average", national_avg)
            with cm4:
                comparison_metric_card(
                    "Yield Improvement", improvement_pct, unit="%",
                    delta=improvement_pct, delta_label="%"
                )

            # More metrics
            cm5, cm6, cm7, cm8 = st.columns(4)
            with cm5:
                rank_str = f"#{crop_rank} of {total_crops_in_state}" if crop_rank > 0 else "N/A"
                metric_card("Crop Rank (State)", rank_str, "🏅")
            with cm6:
                metric_card("Area Used", f"{area} {area_unit}", "📐")
            with cm7:
                metric_card("Inference Time", f"{inference_ms:.0f} ms", "⚡")
            with cm8:
                metric_card("Weather Source", "Live" if is_live_weather else "Historical", "🌦")

            # Prediction Metadata Table
            st.markdown("<br>", unsafe_allow_html=True)
            col_meta1, col_meta2 = st.columns(2)
            with col_meta1:
                executive_metric_table([
                    ("Prediction ID", prediction_id),
                    ("Timestamp", prediction_ts),
                    ("Model Used", "CatBoost Regressor"),
                    ("Model Version", "v2.0.0"),
                    ("Target Transform", "log1p → expm1"),
                ])
            with col_meta2:
                executive_metric_table([
                    ("Crop Year", str(year)),
                    ("Season", season),
                    ("Location", f"{district.title()}, {state.title()}"),
                    ("Coordinates", f"{lat:.4f}°N, {lon:.4f}°E"),
                    ("Features Used", "60"),
                ])

            # ══════════════════════════════════════════════════
            # CHARTS DASHBOARD
            # ══════════════════════════════════════════════════

            # Weather KPIs
            section_header("🌦 Environmental Profile")
            w1, w2, w3, w4 = st.columns(4)
            with w1:
                metric_card("Avg Temperature", f"{weather_summary['temperature_2m_mean']:.1f} °C", "🌡")
            with w2:
                metric_card("Total Rainfall", f"{weather_summary['precipitation_sum']:.1f} mm", "🌧")
            with w3:
                metric_card("Avg Humidity", f"{weather_summary['relative_humidity_2m_mean']:.0f}%", "💧")
            with w4:
                metric_card("Wind Speed", f"{weather_summary['wind_speed_10m_mean']:.1f} m/s", "🌬")

            # Chart 1 + Chart 8: Grouped Bar + Confidence Gauge
            section_header("📈 Yield Analysis")
            ch1, ch8 = st.columns([2, 1])
            with ch1:
                fig1 = yield_comparison_grouped_bar(yield_val_ha, district_avg, state_avg, national_avg)
                st.plotly_chart(fig1, use_container_width=True)
            with ch8:
                fig8 = confidence_gauge(confidence_score, confidence_label)
                st.plotly_chart(fig8, use_container_width=True)

            # Chart 2: Historical Trend
            try:
                hist_trend = df_clean[
                    (df_clean['district_name'] == district.lower()) &
                    (df_clean['crop_lower'] == crop.lower())
                ].groupby('crop_year')['crop_yield'].mean().dropna().reset_index()

                if hist_trend.empty:
                    hist_trend = df_clean[
                        (df_clean['state_name'] == state.lower()) &
                        (df_clean['crop_lower'] == crop.lower())
                    ].groupby('crop_year')['crop_yield'].mean().dropna().reset_index()

                if not hist_trend.empty:
                    fig2 = yield_trend_line(hist_trend, year, yield_val_ha)
                    st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                pass

            # Chart 3 + Chart 4: District Ranking + Crop Comparison
            ch3, ch4 = st.columns(2)

            with ch3:
                try:
                    district_data = df_clean[
                        (df_clean['state_name'] == state.lower()) &
                        (df_clean['crop_lower'] == crop.lower())
                    ].groupby('district_name')['crop_yield'].mean().dropna().reset_index()
                    if not district_data.empty:
                        fig3 = district_ranking_bar(district_data, district)
                        st.plotly_chart(fig3, use_container_width=True)
                except Exception:
                    pass

            with ch4:
                try:
                    top_crops = df_clean[df_clean['state_name'] == state.lower()].groupby('crop_lower')['crop_yield'].mean().dropna()
                    top_crops = top_crops.replace([np.inf, -np.inf], np.nan).dropna()
                    # Get top 5 + selected crop
                    top5 = top_crops.nlargest(5).reset_index()
                    top5 = top5.rename(columns={'crop_lower': 'crop'})
                    if crop.lower() not in top5['crop'].values:
                        sel_crop_yield = top_crops.get(crop.lower(), 0)
                        if sel_crop_yield > 0:
                            new_row = pd.DataFrame({'crop': [crop.lower()], 'crop_yield': [sel_crop_yield]})
                            top5 = pd.concat([top5, new_row], ignore_index=True)
                    if not top5.empty:
                        fig4 = crop_comparison_bar(top5, crop)
                        st.plotly_chart(fig4, use_container_width=True)
                except Exception:
                    pass

            # Chart 5 + Chart 9: Season Donut + Yield Distribution
            ch5, ch9 = st.columns(2)

            with ch5:
                try:
                    season_data = df_clean[df_clean['state_name'] == state.lower()].groupby('season')['crop_yield'].mean().dropna().reset_index()
                    if not season_data.empty:
                        fig5 = season_donut(season_data)
                        st.plotly_chart(fig5, use_container_width=True)
                except Exception:
                    pass

            with ch9:
                try:
                    crop_dist = df_clean[
                        (df_clean['state_name'] == state.lower()) &
                        (df_clean['crop_lower'] == crop.lower())
                    ]
                    if not crop_dist.empty:
                        fig9 = production_histogram(crop_dist, crop)
                        st.plotly_chart(fig9, use_container_width=True)
                except Exception:
                    pass

            # Chart 7: Feature Importance (SHAP)
            section_header("🧠 Why This Prediction?",
                           "Feature importance shows how each input variable influenced the final yield prediction.")
            try:
                fig7 = feature_importance_bar(feature_names, shap_values)
                st.plotly_chart(fig7, use_container_width=True)
            except Exception:
                pass  # Silently skip if SHAP visualization fails

            # Chart 10: Heatmap — State × Season
            section_header("🗺 Yield Patterns")
            ch10, ch_map = st.columns(2)

            with ch10:
                try:
                    heatmap_data = df_clean[df_clean['state_name'] == state.lower()].copy()
                    if not heatmap_data.empty:
                        top_districts = (
                            heatmap_data.groupby('district_name')['crop_yield']
                            .mean().dropna().nlargest(12).index.tolist()
                        )
                        heatmap_filtered = heatmap_data[heatmap_data['district_name'].isin(top_districts)]
                        pivot = heatmap_filtered.pivot_table(
                            values='crop_yield', index='district_name', columns='season', aggfunc='mean'
                        ).fillna(0)
                        pivot = pivot.replace([np.inf, -np.inf], 0)
                        pivot.index = pivot.index.str.title()
                        pivot.columns = pivot.columns.str.title()
                        if not pivot.empty:
                            fig10 = yield_heatmap(pivot)
                            st.plotly_chart(fig10, use_container_width=True)
                except Exception:
                    pass

            with ch_map:
                try:
                    map_data = df_clean.groupby('state_name').agg({
                        'crop_yield': 'mean', 'latitude': 'mean', 'longitude': 'mean'
                    }).reset_index()
                    map_data['crop_yield'] = map_data['crop_yield'].replace([np.inf, -np.inf], np.nan).fillna(0)
                    if not map_data.empty:
                        fig_map = india_map(map_data, selected_state=state)
                        st.plotly_chart(fig_map, use_container_width=True)
                except Exception:
                    pass

            # ── Recommendations ──
            section_header("💡 Agronomic Intelligence",
                           "Data-driven recommendations based on weather, yield, and historical patterns")

            recs = generate_recommendations(
                state, district, crop, season, weather_summary, yield_val_ha, district_avg
            )

            # Group by section
            sections_order = ["Agronomic Advice", "Climate Risk", "Pest & Disease Risk", "Irrigation", "Best Practices"]
            for sec in sections_order:
                sec_recs = [r for r in recs if r.get('section', '') == sec]
                if sec_recs:
                    st.markdown(f'<div class="rec-section-header">📌 {sec}</div>', unsafe_allow_html=True)
                    for r in sec_recs:
                        recommendation_card(r['title'], r['text'], r['type'], r.get('rationale', ''))

            # ── Downloads ──
            st.markdown('<br>', unsafe_allow_html=True)
            section_header("📥 Export Results")

            exec_metrics = {
                'district_avg': district_avg,
                'state_avg': state_avg,
                'national_avg': national_avg,
                'diff_vs_hist': diff_vs_hist,
                'improvement_pct': improvement_pct,
                'confidence': confidence_label,
                'prediction_id': prediction_id,
                'weather_source': weather_source,
                'inference_ms': inference_ms,
            }

            inputs_for_pdf = {
                "state": state, "district": district, "crop_year": year,
                "season": season, "area": area, "area_unit": area_unit,
            }
            predictions_for_pdf = [{
                "Crop": crop, "Yield": yield_val_ha, "Production": total_production
            }]

            pdf_bytes = generate_pdf_report(
                inputs_for_pdf, predictions_for_pdf, weather_summary, recs, exec_metrics
            )

            dl1, dl2, _ = st.columns([1, 1, 2])
            with dl1:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"AgriAI_Report_{district}_{crop}_{prediction_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            with dl2:
                csv_data = pd.DataFrame([st.session_state.history[0]]).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_data,
                    file_name=f"AgriAI_Data_{district}_{crop}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Prediction could not be completed. Please verify your inputs and try again.")
