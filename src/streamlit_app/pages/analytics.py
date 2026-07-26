import streamlit as st
from src.services.data_service import get_dataframe
from src.streamlit_app.components.cards import metric_card
from src.streamlit_app.components.charts import (
    yield_comparison_bar, state_area_bar, 
    production_pie, historical_trend_line, state_choropleth
)

def render():
    """Render the Analytics page."""
    st.markdown('<div class="section-header">📊 Business Intelligence Analytics</div>', unsafe_allow_html=True)
    
    with st.spinner("Loading analytics data..."):
        df = get_dataframe()
        
    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Records", f"{len(df):,}", "📂")
    with c2:
        metric_card("States Covered", f"{df['state_name'].nunique()}", "🗺")
    with c3:
        metric_card("Districts Covered", f"{df['district_name'].nunique()}", "📍")
    with c4:
        metric_card("National Mean", f"{df['crop_yield'].mean():.2f} T/Ha", "🌾")
        
    st.markdown('<div class="section-header">Geographical Distribution</div>', unsafe_allow_html=True)
    
    # Map
    state_yield = df.groupby("state_name").agg({"crop_yield": "mean", "latitude": "mean", "longitude": "mean"}).reset_index()
    # Provide a default district for hover to prevent key errors
    state_yield["district_name"] = "All Districts"
    
    c_map, c_pie = st.columns([2, 1])
    with c_map:
        st.markdown("**National Yield Map**")
        fig_map = state_choropleth(state_yield)
        st.plotly_chart(fig_map, use_container_width=True)
        
    with c_pie:
        st.markdown("**Seasonal Breakdown**")
        season_avg = df.groupby("season")["crop_yield"].mean().reset_index()
        fig_pie = production_pie(season_avg)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    st.markdown('<div class="section-header">Performance Metrics</div>', unsafe_allow_html=True)
    
    c_bar1, c_bar2 = st.columns(2)
    
    with c_bar1:
        st.markdown("**Top 10 States by Cultivated Area**")
        state_area = df.groupby("state_name")["area"].sum().nlargest(10).reset_index()
        state_area['state_name'] = state_area['state_name'].str.title()
        fig_area = state_area_bar(state_area)
        st.plotly_chart(fig_area, use_container_width=True)
        
    with c_bar2:
        st.markdown("**Top 10 Crops by Average Yield**")
        crop_yield = df.groupby("crop")["crop_yield"].mean().nlargest(10).reset_index()
        crop_yield['crop'] = crop_yield['crop'].str.title()
        fig_yield = yield_comparison_bar(crop_yield)
        st.plotly_chart(fig_yield, use_container_width=True)
        
    st.markdown('<div class="section-header">Historical Trends</div>', unsafe_allow_html=True)
    
    trend_df = df.groupby("crop_year")["crop_yield"].mean().reset_index()
    fig_trend = historical_trend_line(trend_df)
    st.plotly_chart(fig_trend, use_container_width=True)
