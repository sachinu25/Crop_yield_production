"""
Premium chart library — light-theme-first Plotly charts.

All chart functions return Plotly figures. No fake data.
Every chart uses real data passed from the calling page.
"""

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np

# ── Premium Light Theme ──
LIGHT_THEME = go.layout.Template(
    layout=go.Layout(
        plot_bgcolor='#ffffff',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, -apple-system, sans-serif", color="#475569", size=12),
        title=dict(font=dict(color="#0f172a", size=15, weight="bold"), x=0, xanchor="left"),
        xaxis=dict(
            showgrid=True, gridcolor='#f1f5f9', gridwidth=1,
            zeroline=False, linecolor='#e2e8f0', linewidth=1,
            tickfont=dict(size=11, color="#64748b")
        ),
        yaxis=dict(
            showgrid=True, gridcolor='#f1f5f9', gridwidth=1,
            zeroline=False, linecolor='#e2e8f0', linewidth=1,
            tickfont=dict(size=11, color="#64748b")
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11, color="#64748b"),
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        hoverlabel=dict(
            bgcolor="#0f172a", font_color="#ffffff", font_size=12,
            bordercolor="#334155"
        ),
        colorway=[
            '#059669', '#2563eb', '#7c3aed', '#d97706', '#dc2626',
            '#0d9488', '#4f46e5', '#be185d', '#0369a1', '#65a30d'
        ]
    )
)

pio.templates["premium_light"] = LIGHT_THEME
pio.templates.default = "premium_light"

# ── Color Palettes ──
COLORS = {
    'primary': '#059669',
    'secondary': '#2563eb',
    'tertiary': '#7c3aed',
    'quaternary': '#d97706',
    'danger': '#dc2626',
    'teal': '#0d9488',
    'positive': '#059669',
    'negative': '#dc2626',
    'neutral': '#64748b',
}

BAR_COLORS = ['#059669', '#2563eb', '#7c3aed', '#d97706']


# ═══════════════════════════════════════════════════════════
# PREDICTION PAGE CHARTS
# ═══════════════════════════════════════════════════════════

def yield_comparison_grouped_bar(predicted: float, district_avg: float, state_avg: float, national_avg: float):
    """Chart 1: Predicted vs District vs State vs National Average — Grouped Bar."""
    data = pd.DataFrame({
        'Category': ['Predicted Yield', 'District Average', 'State Average', 'National Average'],
        'Yield (T/Ha)': [predicted, district_avg, state_avg, national_avg]
    })
    fig = go.Figure()
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], COLORS['quaternary']]
    for i, row in data.iterrows():
        fig.add_trace(go.Bar(
            x=[row['Category']], y=[row['Yield (T/Ha)']],
            name=row['Category'], marker_color=colors[i],
            text=[f"{row['Yield (T/Ha)']:.2f}"], textposition='outside',
            textfont=dict(size=12, weight='bold'),
            width=0.5
        ))
    fig.update_layout(
        title="Yield Comparison",
        yaxis_title="Yield (T/Ha)",
        showlegend=False,
        bargap=0.3,
        height=380,
    )
    return fig


def yield_trend_line(hist_df: pd.DataFrame, pred_year: int, pred_yield: float):
    """Chart 2: Historical yield trend (2001-2020) + prediction point — Smooth Line."""
    fig = go.Figure()
    # Historical line
    fig.add_trace(go.Scatter(
        x=hist_df['crop_year'], y=hist_df['crop_yield'],
        mode='lines+markers', name='Historical',
        line=dict(color=COLORS['secondary'], width=2.5, shape='spline'),
        marker=dict(size=5, color=COLORS['secondary']),
    ))
    # Prediction point
    fig.add_trace(go.Scatter(
        x=[pred_year], y=[pred_yield],
        mode='markers', name='Predicted',
        marker=dict(size=14, color=COLORS['primary'], symbol='star',
                    line=dict(width=2, color='#ffffff')),
    ))
    fig.update_layout(
        title="Historical Yield Trend + Prediction",
        xaxis_title="Year",
        yaxis_title="Yield (T/Ha)",
        height=380,
    )
    return fig


def district_ranking_bar(df: pd.DataFrame, selected_district: str):
    """Chart 3: Top districts for a crop in state — Horizontal Bar with highlight."""
    df = df.copy()
    df['color'] = df['district_name'].apply(
        lambda x: COLORS['primary'] if x.lower() == selected_district.lower() else '#cbd5e1'
    )
    df = df.sort_values('crop_yield', ascending=True).tail(15)
    fig = go.Figure(go.Bar(
        x=df['crop_yield'], y=df['district_name'].str.title(),
        orientation='h', marker_color=df['color'],
        text=df['crop_yield'].round(2), textposition='outside',
        textfont=dict(size=10),
    ))
    fig.update_layout(
        title="District Ranking by Yield",
        xaxis_title="Avg Yield (T/Ha)",
        height=max(350, len(df) * 28),
        yaxis=dict(categoryorder='total ascending'),
        showlegend=False,
    )
    return fig


def crop_comparison_bar(df: pd.DataFrame, selected_crop: str):
    """Chart 4: Selected crop vs top 5 crops in same state."""
    df = df.copy()
    df['color'] = df['crop'].apply(
        lambda x: COLORS['primary'] if x.lower() == selected_crop.lower() else COLORS['secondary']
    )
    fig = go.Figure(go.Bar(
        x=df['crop'].str.title(), y=df['crop_yield'],
        marker_color=df['color'],
        text=df['crop_yield'].round(2), textposition='outside',
        textfont=dict(size=11, weight='bold'),
    ))
    fig.update_layout(
        title="Crop Comparison (Same State)",
        yaxis_title="Avg Yield (T/Ha)",
        showlegend=False,
        height=380,
    )
    return fig


def season_donut(df: pd.DataFrame):
    """Chart 5: Season yield distribution — Donut Chart."""
    colors = ['#059669', '#2563eb', '#7c3aed', '#d97706', '#0d9488', '#dc2626']
    fig = px.pie(
        df, values='crop_yield', names='season', hole=0.55,
        color_discrete_sequence=colors,
    )
    fig.update_traces(
        textposition='inside', textinfo='percent+label',
        textfont=dict(size=11),
    )
    fig.update_layout(
        title="Yield by Season",
        showlegend=False,
        height=380,
    )
    return fig


# Human-readable feature name mapping (shared by SHAP charts)
FEATURE_LABELS = {
        'crop_year': 'Crop Year',
        'log_area': 'Cultivated Area (log)',
        'log_precip': 'Rainfall (log)',
        'temperature_2m_mean': 'Temperature',
        'precipitation_sum': 'Rainfall',
        'relative_humidity_2m_mean': 'Humidity',
        'wind_speed_10m_mean': 'Wind Speed',
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'temp_precip_ratio': 'Temp/Rainfall Ratio',
        'precip_per_temp': 'Rainfall/Temp Ratio',
        'temp_x_humidity': 'Temp × Humidity',
        'precip_x_humidity': 'Rainfall × Humidity',
        'wind_x_temp': 'Wind × Temperature',
        'humidity_per_wind': 'Humidity/Wind Ratio',
        'is_drought': 'Drought Indicator',
        'is_extreme_heat': 'Extreme Heat Indicator',
        'is_high_humidity': 'High Humidity Indicator',
        'gdd_proxy': 'Growing Degree Days',
        'gdd_x_precip': 'GDD × Rainfall',
        'temp_squared': 'Temperature²',
        'precip_squared': 'Rainfall²',
        'years_since_2001': 'Years Since 2001',
        'year_norm_sq': 'Year Progress²',
        'season_sin': 'Season (Cyclic Sin)',
        'season_cos': 'Season (Cyclic Cos)',
        'exp_mean_state_crop': 'State-Crop Avg Yield',
        'roll3_mean_state_crop': 'State-Crop 3yr Rolling',
        'ewma_state_crop': 'State-Crop Weighted Avg',
        'exp_mean_district_crop': 'District-Crop Avg Yield',
        'roll3_mean_district_crop': 'District-Crop 3yr Rolling',
        'ewma_district_crop': 'District-Crop Weighted Avg',
        'exp_mean_crop': 'Crop National Avg',
        'roll3_mean_crop': 'Crop 3yr Rolling',
        'ewma_crop': 'Crop Weighted Avg',
        'exp_mean_state_name': 'State Avg Yield',
        'roll3_mean_state_name': 'State 3yr Rolling',
        'ewma_state_name': 'State Weighted Avg',
        'yield_trend_sc': 'Yield Trend (State-Crop)',
        'temp_anomaly': 'Temperature Anomaly',
        'precip_anomaly': 'Rainfall Anomaly',
        'climate_pca_1': 'Climate Component 1',
        'climate_pca_2': 'Climate Component 2',
        'freq_district_name': 'District Data Frequency',
        'freq_state_name': 'State Data Frequency',
        'freq_crop': 'Crop Data Frequency',
        'freq_season': 'Season Data Frequency',
        'looe_crop': 'Crop Encoded Yield',
        'looe_state_name': 'State Encoded Yield',
        'looe_district_name': 'District Encoded Yield',
        'looe_state_crop': 'State-Crop Encoded Yield',
        'looe_district_crop': 'District-Crop Encoded Yield',
        'season': 'Season',
        'crop': 'Crop Type',
        'state_name': 'State',
        'crop_season': 'Crop-Season',
        'state_crop': 'State-Crop',
        'geo_cluster_10': 'Geo Cluster (k=10)',
        'geo_cluster_30': 'Geo Cluster (k=30)',
        'geo_cluster_50': 'Geo Cluster (k=50)',
    }


def feature_importance_bar(feature_names: list, shap_values: list, top_n: int = 12):
    """Chart 7: Feature importance — Horizontal bar with English labels & color coding."""
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_values,
        'AbsImpact': np.abs(shap_values)
    })
    top = shap_df.nlargest(top_n, 'AbsImpact').sort_values('AbsImpact', ascending=True)
    top['Label'] = top['Feature'].map(FEATURE_LABELS).fillna(top['Feature'])
    top['Color'] = top['SHAP Value'].apply(lambda v: COLORS['positive'] if v > 0 else COLORS['negative'])

    fig = go.Figure(go.Bar(
        x=top['SHAP Value'], y=top['Label'],
        orientation='h', marker_color=top['Color'],
        text=top['SHAP Value'].round(3), textposition='outside',
        textfont=dict(size=10),
    ))
    fig.update_layout(
        title="Feature Importance (SHAP Values)",
        xaxis_title="Impact on Prediction (log scale)",
        height=max(350, top_n * 32),
        showlegend=False,
    )
    return fig


def shap_waterfall(feature_names: list, shap_values: list, base_value: float, top_n: int = 10):
    """SHAP waterfall: from the model's average output to this prediction, step by step.

    Values are in log1p(yield) space (the model's training target); the remaining
    features are grouped into one 'Other features' step so the bars still sum exactly.
    """
    sv = np.asarray(shap_values, dtype=float)
    order = np.argsort(-np.abs(sv))
    top_idx, rest_idx = order[:top_n], order[top_n:]
    labels = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in top_idx]
    values = [float(sv[i]) for i in top_idx]
    if len(rest_idx):
        labels.append(f"Other {len(rest_idx)} features")
        values.append(float(sv[rest_idx].sum()))
    final = base_value + float(sv.sum())

    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=["absolute"] + ["relative"] * len(values) + ["total"],
        y=["Model average (log)"] + labels + ["This prediction (log)"],
        x=[base_value] + values + [final],
        text=[f"{base_value:.3f}"] + [f"{v:+.3f}" for v in values] + [f"{final:.3f}"],
        textposition="outside", cliponaxis=False,
        increasing=dict(marker=dict(color=COLORS['positive'])),
        decreasing=dict(marker=dict(color=COLORS['negative'])),
        totals=dict(marker=dict(color=COLORS['secondary'])),
        connector=dict(line=dict(color="#cbd5e1", width=1)),
    ))
    fig.update_layout(
        title="SHAP Waterfall — How Each Feature Moved This Prediction",
        xaxis_title="log(1 + yield)  ·  green raises, red lowers the prediction",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=40, r=70, t=50, b=40),
        height=max(380, (len(values) + 2) * 34),
        showlegend=False,
    )
    return fig


def benchmark_r2_bar(benchmark: dict):
    """Grouped bar: R² of each model before vs after the training-derived quality filter."""
    names = list(benchmark.keys())
    fig = go.Figure()
    for key, label, color in [("full", "All test rows", COLORS['neutral']),
                              ("quality_filtered", "Quality-filtered", COLORS['primary'])]:
        vals = [benchmark[n][key]["r2"] for n in names]
        fig.add_trace(go.Bar(x=names, y=vals, name=label, marker_color=color,
                             text=[f"{v:.3f}" for v in vals], textposition="outside"))
    fig.update_layout(
        title="Out-of-Time R² (2018–2020): Before vs After Quality Filter",
        yaxis=dict(title="R²", range=[0, 1.05]),
        barmode="group", height=400,
    )
    return fig


def confidence_gauge(confidence_score: float, confidence_label: str):
    """Chart 8: Confidence level — Gauge chart."""
    color_map = {
        'Very High': '#047857', 'High': '#059669',
        'Medium': '#d97706', 'Low': '#dc2626'
    }
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence_score,
        number=dict(suffix="%", font=dict(size=28, weight='bold', color='#0f172a')),
        title=dict(text=f"Confidence: {confidence_label}", font=dict(size=14, color='#475569')),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor='#94a3b8', tickwidth=1),
            bar=dict(color=color_map.get(confidence_label, '#059669'), thickness=0.7),
            bgcolor='#f1f5f9',
            borderwidth=0,
            steps=[
                dict(range=[0, 30], color='#fee2e2'),
                dict(range=[30, 60], color='#fef3c7'),
                dict(range=[60, 80], color='#d1fae5'),
                dict(range=[80, 100], color='#a7f3d0'),
            ],
            threshold=dict(
                line=dict(color=color_map.get(confidence_label, '#059669'), width=3),
                thickness=0.8, value=confidence_score
            ),
        )
    ))
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=60, b=20))
    return fig


def production_histogram(df: pd.DataFrame, crop: str):
    """Chart 9: Production distribution — Histogram."""
    fig = px.histogram(
        df, x='crop_yield', nbins=30,
        color_discrete_sequence=[COLORS['secondary']],
        labels={'crop_yield': 'Yield (T/Ha)', 'count': 'Frequency'},
    )
    fig.update_layout(
        title=f"Yield Distribution — {crop.title()}",
        yaxis_title="Frequency",
        height=350,
        bargap=0.05,
    )
    fig.update_traces(marker_line_color='#ffffff', marker_line_width=0.5)
    return fig


def yield_heatmap(pivot_df: pd.DataFrame):
    """Chart 10: State × Season historical yield — Heatmap."""
    fig = go.Figure(go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns.tolist(),
        y=pivot_df.index.tolist(),
        colorscale=[
            [0, '#f0fdf4'], [0.25, '#bbf7d0'], [0.5, '#4ade80'],
            [0.75, '#16a34a'], [1, '#14532d']
        ],
        hovertemplate='State: %{y}<br>Season: %{x}<br>Yield: %{z:.2f} T/Ha<extra></extra>',
        colorbar=dict(title="T/Ha", title_font=dict(size=11)),
    ))
    fig.update_layout(
        title="Yield Heatmap — State × Season",
        xaxis_title="Season",
        yaxis_title="State",
        height=max(400, len(pivot_df) * 22),
        yaxis=dict(tickfont=dict(size=10)),
        xaxis=dict(tickfont=dict(size=11)),
    )
    return fig


def india_map(df: pd.DataFrame, selected_state: str = None, selected_district: str = None):
    """India map — scatter_geo with state highlight."""
    df = df.copy()
    if selected_state:
        df['is_selected'] = df['state_name'].str.lower() == selected_state.lower()
        df['marker_size'] = df['is_selected'].apply(lambda x: 18 if x else 8)
        df['opacity'] = df['is_selected'].apply(lambda x: 1.0 if x else 0.5)
    else:
        df['marker_size'] = 10
        df['opacity'] = 0.7

    fig = px.scatter_geo(
        df, lat='latitude', lon='longitude',
        size='marker_size', color='crop_yield',
        hover_name='state_name',
        hover_data={'crop_yield': ':.2f', 'marker_size': False, 'opacity': False},
        color_continuous_scale=[
            [0, '#d1fae5'], [0.3, '#6ee7b7'], [0.6, '#34d399'], [1, '#047857']
        ],
        projection='natural earth',
        labels={'crop_yield': 'Yield (T/Ha)'},
    )
    fig.update_geos(
        visible=True,
        showcountries=True, countrycolor='#cbd5e1',
        showcoastlines=True, coastlinecolor='#cbd5e1',
        showland=True, landcolor='#f8fafc',
        showocean=True, oceancolor='#eff6ff',
        fitbounds='locations',
        bgcolor='rgba(0,0,0,0)',
    )
    fig.update_layout(
        margin=dict(r=0, t=30, l=0, b=0),
        height=450,
        title="India — Agricultural Yield Map",
        coloraxis_colorbar=dict(title="T/Ha", title_font=dict(size=11)),
    )
    return fig


# ═══════════════════════════════════════════════════════════
# ANALYTICS PAGE CHARTS
# ═══════════════════════════════════════════════════════════

def district_climate_yield_chart(yearly: pd.DataFrame, crop: str = None):
    """Yearly rainfall anomaly (bars), temperature anomaly (line) and optional crop yield (right axis)."""
    fig = go.Figure()
    rain_colors = [COLORS['secondary'] if v >= 0 else COLORS['quaternary'] for v in yearly['rain_anomaly_pct']]
    fig.add_trace(go.Bar(x=yearly['crop_year'], y=yearly['rain_anomaly_pct'], name='Rainfall anomaly (%)',
                         marker_color=rain_colors, opacity=0.75))
    fig.add_trace(go.Scatter(x=yearly['crop_year'], y=yearly['temp_anomaly'] * 10, name='Temp anomaly (°C × 10)',
                             mode='lines+markers', line=dict(color=COLORS['danger'], width=2),
                             customdata=yearly['temp_anomaly'],
                             hovertemplate='%{x}: %{customdata:+.2f} °C<extra></extra>'))
    if crop and 'crop_yield' in yearly and yearly['crop_yield'].notna().any():
        fig.add_trace(go.Scatter(x=yearly['crop_year'], y=yearly['crop_yield'], name=f'{crop.title()} yield (T/Ha)',
                                 mode='lines+markers', yaxis='y2', line=dict(color=COLORS['primary'], width=3)))
    fig.update_layout(
        title="Weather Anomalies and Yield by Year",
        xaxis=dict(title="Year", dtick=2),
        yaxis=dict(title="Anomaly"),
        yaxis2=dict(title="Yield (T/Ha)", overlaying='y', side='right', showgrid=False),
        height=420, barmode='relative',
    )
    return fig


def weather_trend_chart(env_data: pd.DataFrame):
    """Multi-line weather trend chart."""
    fig = go.Figure()
    metrics = [
        ('temperature_2m_mean', 'Temperature (°C)', '#dc2626'),
        ('precipitation_sum', 'Rainfall (mm)', '#2563eb'),
        ('relative_humidity_2m_mean', 'Humidity (%)', '#059669'),
        ('wind_speed_10m_mean', 'Wind (m/s)', '#7c3aed'),
    ]
    for col, label, color in metrics:
        if col in env_data.columns:
            fig.add_trace(go.Scatter(
                x=env_data['date'], y=env_data[col],
                mode='lines', name=label,
                line=dict(color=color, width=2),
            ))
    fig.update_layout(
        title="Daily Weather Parameters",
        xaxis_title="Date", yaxis_title="Value",
        height=400,
    )
    return fig


def yield_comparison_bar(df: pd.DataFrame):
    """Top crops by yield — horizontal bar."""
    fig = go.Figure(go.Bar(
        x=df['crop_yield'], y=df['crop'],
        orientation='h',
        marker=dict(
            color=df['crop_yield'],
            colorscale=[[0, '#d1fae5'], [0.5, '#34d399'], [1, '#047857']],
        ),
        text=df['crop_yield'].round(2), textposition='outside',
        textfont=dict(size=10),
    ))
    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),
        height=380,
        showlegend=False,
    )
    return fig


def state_area_bar(df: pd.DataFrame):
    """Top states by area — horizontal bar."""
    fig = go.Figure(go.Bar(
        x=df['area'], y=df['state_name'],
        orientation='h',
        marker=dict(
            color=df['area'],
            colorscale=[[0, '#dbeafe'], [0.5, '#60a5fa'], [1, '#1d4ed8']],
        ),
        text=df['area'].apply(lambda x: f"{x:,.0f}"), textposition='outside',
        textfont=dict(size=10),
    ))
    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),
        height=380,
        showlegend=False,
    )
    return fig


def production_pie(df: pd.DataFrame):
    """Seasonal yield share — donut."""
    return season_donut(df)


def historical_trend_line(df: pd.DataFrame):
    """Year-over-year national trend."""
    fig = go.Figure(go.Scatter(
        x=df['crop_year'], y=df['crop_yield'],
        mode='lines+markers', name='National Average',
        line=dict(color=COLORS['primary'], width=2.5, shape='spline'),
        marker=dict(size=6, color=COLORS['primary']),
        fill='tozeroy', fillcolor='rgba(5, 150, 105, 0.06)',
    ))
    fig.update_layout(
        title="National Average Yield Trend",
        xaxis_title="Year", yaxis_title="Avg Yield (T/Ha)",
        height=380,
    )
    return fig


def state_choropleth(df: pd.DataFrame):
    """State-level map — scatter_geo."""
    return india_map(df)
