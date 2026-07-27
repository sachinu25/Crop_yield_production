"""
Analytics Page — Business Intelligence Dashboard.

Interactive BI dashboard with filters, KPIs, and Plotly charts.
All data from the real dataset. No fake or placeholder content.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.services.data_service import get_dataframe, get_dropdown_options
from src.streamlit_app.components.cards import metric_card, section_header
from src.streamlit_app.components.charts import (
    yield_comparison_bar, state_area_bar, season_donut,
    historical_trend_line, india_map, yield_heatmap,
)


@st.cache_data(ttl=3600)
def _load_analytics_data():
    """Cache the dataframe for analytics."""
    return get_dataframe()


def render():
    """Render the Analytics page."""
    section_header("📊 Business Intelligence Dashboard",
                   "Interactive analytics across 20 years of agricultural data")

    df = _load_analytics_data()
    options = get_dropdown_options(title_case=True)

    # ── Filters ──
    with st.container(border=True):
        st.markdown("**🔍 Filters**")
        f1, f2, f3, f4 = st.columns(4)

        with f1:
            sel_states = st.multiselect(
                "States", options["state_list"],
                placeholder="All States", key="ana_states"
            )
        with f2:
            sel_seasons = st.multiselect(
                "Seasons", options["season_list"],
                placeholder="All Seasons", key="ana_seasons"
            )
        with f3:
            all_years = sorted(df['crop_year'].unique())
            year_range = st.select_slider(
                "Year Range",
                options=all_years,
                value=(all_years[0], all_years[-1]),
                key="ana_years"
            )
        with f4:
            sel_crops = st.multiselect(
                "Crops", sorted([c.title() for c in options["crop_list"][:50]]),
                placeholder="All Crops", key="ana_crops"
            )

    # Apply filters
    filtered = df.copy()
    if sel_states:
        filtered = filtered[filtered['state_name'].isin([s.lower() for s in sel_states])]
    if sel_seasons:
        filtered = filtered[filtered['season'].isin([s.lower() for s in sel_seasons])]
    if sel_crops:
        filtered = filtered[filtered['crop'].str.lower().isin([c.lower() for c in sel_crops])]
    filtered = filtered[
        (filtered['crop_year'] >= year_range[0]) &
        (filtered['crop_year'] <= year_range[1])
    ]

    # Guard against inf values for mean computation
    clean_yield = filtered['crop_yield'].replace([np.inf, -np.inf], np.nan).dropna()

    # ── KPI Row ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Records", f"{len(filtered):,}", "📂")
    with c2:
        metric_card("States", f"{filtered['state_name'].nunique()}", "🗺")
    with c3:
        metric_card("Districts", f"{filtered['district_name'].nunique()}", "📍")
    with c4:
        avg_yield = float(clean_yield.mean()) if len(clean_yield) > 0 else 0
        metric_card("Avg Yield", f"{avg_yield:.2f} T/Ha", "🌾")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        metric_card("Crops", f"{filtered['crop'].nunique()}", "🌿")
    with c6:
        total_area = filtered['area'].sum()
        if total_area > 1e6:
            metric_card("Total Area", f"{total_area/1e6:.1f}M Ha", "📐")
        else:
            metric_card("Total Area", f"{total_area:,.0f} Ha", "📐")
    with c7:
        median_yield = float(clean_yield.median()) if len(clean_yield) > 0 else 0
        metric_card("Median Yield", f"{median_yield:.2f} T/Ha", "📊")
    with c8:
        year_span = f"{int(filtered['crop_year'].min())}–{int(filtered['crop_year'].max())}" if len(filtered) > 0 else "N/A"
        metric_card("Year Range", year_span, "📅")

    # ── Charts Row 1: Map + Season Donut ──
    section_header("🗺 Geographical Distribution")
    c_map, c_pie = st.columns([2, 1])

    with c_map:
        state_yield = filtered.groupby('state_name').agg({
            'crop_yield': 'mean', 'latitude': 'mean', 'longitude': 'mean'
        }).reset_index()
        # Replace inf
        state_yield['crop_yield'] = state_yield['crop_yield'].replace([np.inf, -np.inf], np.nan).fillna(0)
        if not state_yield.empty:
            fig_map = india_map(state_yield)
            st.plotly_chart(fig_map, use_container_width=True)

    with c_pie:
        season_avg = filtered.groupby('season')['crop_yield'].mean().reset_index()
        season_avg['crop_yield'] = season_avg['crop_yield'].replace([np.inf, -np.inf], np.nan).fillna(0)
        season_avg['season'] = season_avg['season'].str.title()
        if not season_avg.empty:
            fig_pie = season_donut(season_avg)
            st.plotly_chart(fig_pie, use_container_width=True)

    # ── Charts Row 2: Top States + Top Crops ──
    section_header("🏆 Performance Metrics")
    c_bar1, c_bar2 = st.columns(2)

    with c_bar1:
        st.markdown("**Top 10 States by Cultivated Area**")
        state_area = filtered.groupby('state_name')['area'].sum().nlargest(10).reset_index()
        state_area['state_name'] = state_area['state_name'].str.title()
        if not state_area.empty:
            fig_area = state_area_bar(state_area)
            st.plotly_chart(fig_area, use_container_width=True)

    with c_bar2:
        st.markdown("**Top 10 Crops by Average Yield**")
        crop_clean = filtered.copy()
        crop_clean['crop_yield'] = crop_clean['crop_yield'].replace([np.inf, -np.inf], np.nan)
        crop_yield = crop_clean.groupby('crop')['crop_yield'].mean().dropna().nlargest(10).reset_index()
        crop_yield['crop'] = crop_yield['crop'].str.title()
        if not crop_yield.empty:
            fig_yield = yield_comparison_bar(crop_yield)
            st.plotly_chart(fig_yield, use_container_width=True)

    # ── Historical Trend ──
    section_header("📈 Historical Trends")
    trend_clean = filtered.copy()
    trend_clean['crop_yield'] = trend_clean['crop_yield'].replace([np.inf, -np.inf], np.nan)
    trend_df = trend_clean.groupby('crop_year')['crop_yield'].mean().dropna().reset_index()
    if not trend_df.empty:
        fig_trend = historical_trend_line(trend_df)
        st.plotly_chart(fig_trend, use_container_width=True)

    # ── Heatmap: Top States × Season ──
    section_header("🔥 Yield Heatmap — State × Season")
    heatmap_clean = filtered.copy()
    heatmap_clean['crop_yield'] = heatmap_clean['crop_yield'].replace([np.inf, -np.inf], np.nan)
    top_states = heatmap_clean.groupby('state_name')['crop_yield'].mean().dropna().nlargest(12).index.tolist()
    heatmap_filtered = heatmap_clean[heatmap_clean['state_name'].isin(top_states)]
    if not heatmap_filtered.empty:
        pivot = heatmap_filtered.pivot_table(
            values='crop_yield', index='state_name', columns='season', aggfunc='mean'
        ).fillna(0)
        pivot.index = pivot.index.str.title()
        pivot.columns = pivot.columns.str.title()
        if not pivot.empty:
            fig_heat = yield_heatmap(pivot)
            st.plotly_chart(fig_heat, use_container_width=True)

    # ── Data Explorer ──
    with st.expander("📋 View Filtered Dataset"):
        st.dataframe(
            filtered.head(500),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"Showing first 500 of {len(filtered):,} records")
