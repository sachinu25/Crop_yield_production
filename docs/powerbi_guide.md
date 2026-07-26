# Power BI Direct Connectivity Guide

This document outlines the architecture for connecting Power BI directly to the Indian Agriculture MySQL database, satisfying the requirements for Data Analyst and Business Analyst portfolio presentations.

## 1. Database Connection (DirectQuery Mode)

To ensure real-time analysis and avoid exporting static CSV files (as requested), use Power BI's built-in MySQL connector:

1. Open Power BI Desktop.
2. Click **Get Data** -> **More** -> **MySQL database**.
3. Enter the server details (e.g., `localhost:3306`) and the database name (`indian_agriculture`).
4. Select **DirectQuery** mode (recommended for large datasets if MySQL is hosted, or Import mode for speed locally).
5. Load the Star Schema tables:
   - `dim_state`
   - `dim_district`
   - `dim_season`
   - `dim_crop`
   - `fact_crop_production`
   - `vw_state_annual_summary` (Pre-aggregated View)
   - `vw_crop_national_performance` (Pre-aggregated View)

## 2. Power BI Data Modeling (Star Schema)

In the Power BI Model View, ensure the following 1-to-Many (*) relationships are active:
- `dim_state[state_id]` (1) <---> (*) `dim_district[state_id]`
- `dim_district[district_id]` (1) <---> (*) `fact_crop_production[district_id]`
- `dim_season[season_id]` (1) <---> (*) `fact_crop_production[season_id]`
- `dim_crop[crop_id]` (1) <---> (*) `fact_crop_production[crop_id]`

## 3. Recommended DAX Measures

Create these explicit DAX measures for your KPI Dashboards:

**Total Production:**
```dax
Total Production (Tonnes) = SUM(fact_crop_production[estimated_production_tonnes])
```

**Total Cultivated Area:**
```dax
Total Area (Ha) = SUM(fact_crop_production[area_ha])
```

**Average Yield:**
```dax
Avg Yield (T/Ha) = 
DIVIDE(
    [Total Production (Tonnes)], 
    [Total Area (Ha)], 
    0
)
```

**YoY Production Growth:**
```dax
YoY Growth % = 
VAR CurrentYear = [Total Production (Tonnes)]
VAR PreviousYear = CALCULATE(
    [Total Production (Tonnes)], 
    SAMEPERIODLASTYEAR(dim_date[date]) -- Requires creating a standard dim_date table linked to crop_year
)
RETURN DIVIDE(CurrentYear - PreviousYear, PreviousYear, 0)
```

## 4. Required Dashboard Pages

As per the portfolio requirements, build the following 6 tabs in your Power BI file (`.pbix`):

1. **Executive Dashboard**: High-level KPIs (Total Yield, Active States), Year-over-Year growth, and a map of India colored by total production.
2. **Production Dashboard**: Deep-dive into raw tonnage vs area cultivated, including a scatter plot of Area vs Production with a trendline.
3. **Crop Dashboard**: Drill-down by `crop_category` and `crop_name`. Use a Decomposition Tree visual to break down Category -> Crop -> State.
4. **Weather Dashboard**: Correlation visuals (scatter plots or line-and-clustered-column charts) comparing `temperature_2m_mean` and `precipitation_sum` against Average Yield.
5. **Regional Dashboard**: Matrix visuals detailing State and District level performance, highlighting highest and lowest yielding districts.
6. **Forecast Dashboard**: Use Power BI's native forecasting tool on a line chart of `crop_year` vs `Total Production (Tonnes)` to project the next 3 years.
