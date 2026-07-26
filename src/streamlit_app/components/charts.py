import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Premium dark theme configuration for Plotly
CHART_THEME = {
    'layout': go.Layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color="#9aa0b2"),
        title=dict(font=dict(color="#e8eaed", size=18, weight="bold")),
        xaxis=dict(showgrid=False, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
}

import plotly.io as pio
pio.templates["premium_dark"] = CHART_THEME
pio.templates.default = "premium_dark"

def weather_trend_chart(env_data: pd.DataFrame):
    """Line chart for weather trends."""
    fig = px.line(
        env_data, x="date",
        y=["temperature_2m_mean", "precipitation_sum", "relative_humidity_2m_mean", "wind_speed_10m_mean"],
        labels={"value": "Parameter Value", "variable": "Metric", "date": "Date"},
        color_discrete_sequence=["#f87171", "#60a5fa", "#34d399", "#a78bfa"]
    )
    fig.update_traces(line=dict(width=2.5))
    return fig

def yield_comparison_bar(df: pd.DataFrame):
    """Top 10 crops by yield."""
    fig = px.bar(
        df, x="crop_yield", y="crop", orientation="h",
        color="crop_yield", color_continuous_scale="Viridis",
        labels={"crop_yield": "Avg Yield (T/Ha)", "crop": "Crop"}
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    fig.update_coloraxes(showscale=False)
    return fig

def state_area_bar(df: pd.DataFrame):
    """Top 10 states by area."""
    fig = px.bar(
        df, x="area", y="state_name", orientation="h",
        color="area", color_continuous_scale="Blues",
        labels={"area": "Total Area (Ha)", "state_name": "State"}
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    fig.update_coloraxes(showscale=False)
    return fig

def production_pie(df: pd.DataFrame):
    """Seasonal yield share."""
    fig = px.pie(
        df, values="crop_yield", names="season", hole=0.5,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False)
    return fig

def historical_trend_line(df: pd.DataFrame):
    """Year over year trend."""
    fig = px.line(
        df, x="crop_year", y="crop_yield", markers=True,
        labels={"crop_year": "Year", "crop_yield": "Avg Yield (T/Ha)"},
        color_discrete_sequence=["#34d399"]
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=8, color="#a78bfa"))
    return fig

def state_choropleth(df: pd.DataFrame):
    """India map showing yield by state."""
    # We will use scatter_geo as a lightweight alternative since full geojson for India states
    # might not be available out of the box in basic Plotly without external files.
    # But for a premium look, we'll try an outline approach.
    fig = px.scatter_geo(
        df, lat="latitude", lon="longitude", size="crop_yield", color="crop_yield",
        hover_name="state_name", hover_data=["district_name", "crop_yield"],
        color_continuous_scale="Plasma",
        projection="mercator"
    )
    fig.update_geos(
        fitbounds="locations", visible=False,
        showcountries=True, countrycolor="rgba(255,255,255,0.1)",
        showsubunits=True, subunitcolor="rgba(255,255,255,0.05)",
        bgcolor="rgba(0,0,0,0)"
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig
