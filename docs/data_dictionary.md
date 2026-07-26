# 📖 Data Dictionary — Indian Agriculture & Weather Dataset

This document defines the schema, units, ranges, and descriptions for all attributes in the dataset.

---

## 1. Primary Dataset Fields (`data/processed/final_data_for_web.csv`)

| Field Name | Data Type | Units / Format | Description | Example Values |
|---|---|---|---|---|
| `state_name` | String | Title / Lowercase | Name of the Indian State / Union Territory | `andhra pradesh`, `punjab` |
| `district_name` | String | Title / Lowercase | Name of the District within the State | `anantapur`, `ludhiana` |
| `crop_year` | Integer | YYYY (2001–2020) | Agricultural crop harvest year | `2001`, `2020` |
| `season` | String | Lowercase | Crop season in India | `kharif`, `rabi`, `whole year`, `summer`, `winter`, `autumn` |
| `crop` | String | Lowercase | Crop commodity cultivated | `rice`, `wheat`, `cotton(lint)`, `sugarcane` |
| `area` | Float | Hectares (Ha) | Cultivated land area | `125.50`, `14500.00` |
| `temperature_2m_mean` | Float | Celsius (°C) | Mean daily ambient temperature at 2m height during season | `26.45`, `31.20` |
| `precipitation_sum` | Float | Millimeters (mm) | Total seasonal cumulative precipitation/rainfall | `120.50`, `850.25` |
| `relative_humidity_2m_mean` | Float | Percentage (%) | Mean relative humidity at 2m height during season | `65.2`, `82.0` |
| `wind_speed_10m_mean` | Float | Meters/sec (m/s) | Mean wind speed at 10m height during season | `2.15`, `4.80` |
| `latitude` | Float | Decimal Degrees | Latitude coordinate of district center | `14.7242` |
| `longitude` | Float | Decimal Degrees | Longitude coordinate of district center | `77.4306` |
| `crop_yield` | Float | Tonnes / Hectare | Target variable: Crop output per unit area | `0.285`, `3.450` |

---

## 2. Calculated / Derived Fields (`data/processed/featured_data.csv`)

| Field Name | Data Type | Description | Formula / Rule |
|---|---|---|---|
| `crop_category` | String | Broad taxonomy classification | `Cereal`, `Pulse`, `Oilseed`, `Commercial`, `Fruit/Vegetable`, `Other` |
| `temp_precip_ratio` | Float | Climate stress ratio | `temperature_2m_mean / (precipitation_sum + 1.0)` |
| `is_drought` | Binary (0/1) | Seasonal drought indicator | `1` if `precipitation_sum < 10.0mm`, else `0` |
| `is_extreme_heat` | Binary (0/1) | Heat stress indicator | `1` if `temperature_2m_mean > 32.0°C`, else `0` |
| `log_area` | Float | Natural log of cultivated area | `np.log1p(area)` |

---

## 3. Relational Star Schema Mapping (`data/processed/agriculture.db`)

```
dim_state (state_id PK, state_name)
    └── dim_district (district_id PK, district_name, state_id FK, latitude, longitude)
            └── fact_crop_production (production_id PK, district_id FK, crop_id FK, season_id FK, crop_year, area_ha, yield_tonnes_per_ha, estimated_production_tonnes)
    
dim_crop (crop_id PK, crop_name, crop_category) ───┘
dim_season (season_id PK, season_name) ────────────┘
```
