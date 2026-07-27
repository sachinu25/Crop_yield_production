# 🌾 AI-Powered Crop Yield Prediction System

<div align="center">

<img src="https://img.shields.io/badge/AI%20Powered-Crop%20Yield%20Prediction-2E7D32?style=for-the-badge&logo=leaflet&logoColor=white" alt="AI Powered Crop Yield Prediction" />

<p><strong>Production-ready machine learning platform for predicting Indian crop yield using historical agriculture + weather intelligence.</strong></p>

<p>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/CatBoost-Regressor-FFCC00?style=flat-square" alt="CatBoost" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Enabled-6A1B9A?style=flat-square" alt="Machine Learning" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT License" />
  <img src="https://img.shields.io/github/stars/sachinu25/Crop_yield_production?style=flat-square" alt="GitHub Stars" />
  <img src="https://img.shields.io/badge/Open%20Source-Yes-0A66C2?style=flat-square" alt="Open Source" />
  <img src="https://img.shields.io/badge/Responsive-UI-009688?style=flat-square" alt="Responsive" />
  <img src="https://img.shields.io/badge/Production-Ready-1E88E5?style=flat-square" alt="Production Ready" />
</p>

</div>

---

## 📌 Project Overview

### Problem Statement
Agricultural productivity planning is often reactive and fragmented due to inconsistent access to unified crop, geography, and weather intelligence. Stakeholders need district-level yield forecasting to improve decisions before sowing, during cultivation, and before harvest.

### Solution
This project delivers an end-to-end **AI-powered crop yield prediction system** that combines:
- Historical APY (Area-Production-Yield) records
- District/state geospatial and seasonal context
- Historical weather signals from Open-Meteo
- A finalized CatBoost-based inference pipeline

The system provides both:
- **Streamlit dashboard** (interactive analytics experience)
- **Flask web application** (lightweight production-style app)

### Business Impact
- Better crop planning and resource allocation
- Improved risk assessment for climate variability
- Faster, explainable decisions for yield-sensitive operations
- Strong baseline for AgriTech productization and policy support

### Agriculture Use Cases
| Stakeholder | Value Delivered |
|---|---|
| Government Bodies | District-level planning, subsidy strategy support, regional yield monitoring |
| Farmers & FPOs | Crop selection insights, expected yield benchmarking, season planning |
| Researchers | Reproducible ML pipeline, feature analysis, weather-yield relationship exploration |
| AgriTech Companies | Product integration foundation for advisory, input planning, and intelligence dashboards |

---

## ✨ Features

| Category | Feature | Description |
|---|---|---|
| Prediction | AI Crop Yield Prediction | Predict crop yield (tonnes/hectare) for selected crop, district, season, and area |
| Weather | Historical Weather Integration | Uses Open-Meteo historical signals for climate-aware context |
| ML | Feature Engineering | Uses engineered agri, seasonal, and geo-linked features in finalized pipeline |
| Analytics | Interactive Analytics | Rich exploratory views for comparisons, trends, and distributions |
| Reporting | PDF Reports | Export prediction summaries as shareable reports |
| Data Export | CSV Export | Download predictions/history for external analysis |
| Tracking | Prediction History | Persisted record of prior inferences for quick lookup |
| Explainability | Model Explainability | SHAP-based feature impact analysis for transparent predictions |
| Dashboard | Weather Dashboard | Consolidated weather overview for selected context |
| UI | Responsive UI | Works across desktop/laptop/mobile form factors |
| UI | Light/Dark Mode | Theme support with accessibility-oriented contrast |

---

## 🧰 Technology Stack

| Layer | Tools/Frameworks |
|---|---|
| Frontend | Streamlit, HTML/CSS, Plotly |
| Backend | Flask, Python services |
| Machine Learning | CatBoost Regressor, SHAP |
| Visualization | Plotly, Streamlit chart components |
| Data Handling | Pandas, NumPy, Parquet |
| Configuration | python-dotenv |
| Deployment (App Runtime) | Gunicorn-compatible Flask app, Streamlit runtime |

---

## 🤖 Machine Learning Pipeline

> **Note:** Model training logic is finalized. This repository focuses on production-quality inference, analytics, and presentation.

1. **Dataset Ingestion**
   - APY and related historical agriculture records
   - Weather/context enrichment inputs
2. **Data Cleaning**
   - Missing value handling
   - Type normalization and consistency checks
3. **Feature Engineering**
   - Geo, crop, seasonal, and weather-derived features
4. **Model Training (Finalized)**
   - CatBoost Regressor selected from comparative experiments
5. **Prediction Inference**
   - District/crop/season-aware yield prediction
6. **Visualization Layer**
   - Comparative and temporal charts
7. **Explainability**
   - SHAP insights for local/global feature contribution

---

## 🧪 Feature Engineering Highlights

The pipeline includes engineered features spanning:

- **Geospatial context**: state/district encoded intelligence
- **Crop-level encodings**: crop identity and category sensitivity
- **Seasonal signals**: seasonality and historical performance tendencies
- **Weather-linked attributes**: climate summaries aligned to crop context
- **Area-production relationships**: transformed and normalized agri scale indicators
- **Interaction patterns**: selected combined signals for nonlinear model learning

> Exact feature columns and transformations are available in notebooks and processing scripts.

---

## 📈 Model Performance

From the current project documentation and evaluation artifacts:

| Metric | Value |
|---|---|
| R² Score | ~0.6047 |
| RMSE | ~10.01 |
| MAE | ~1.42 |

### Metric Interpretation
- **R² (Coefficient of Determination):** Explains how much variance in yield is captured by the model. A value near 0.60 indicates meaningful predictive signal in a high-variability domain like agriculture.
- **RMSE (Root Mean Squared Error):** Penalizes larger errors more strongly and reflects prediction spread around true yield.
- **MAE (Mean Absolute Error):** Average absolute prediction deviation; easier to interpret in practical yield terms.

### CV Performance
Cross-validation details should be sourced from model-training artifacts/checkpoints where available. If you add the exact CV table (fold-wise), update this section for publication-grade reporting.

---

## 🗂️ Project Structure

```text
Crop_yield_production/
├── checkpoints/                 # Encoded datasets, train/test split artifacts
├── data/
│   ├── raw/                     # Original APY and source datasets
│   └── processed/               # Clean, model-consumable datasets
├── docs/                        # Supporting documentation and references
├── models/                      # Finalized CatBoost model and metadata
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Model_Training.ipynb
│   ├── 05_Model_Comparison.ipynb
│   ├── 06_SHAP_Analysis.ipynb
│   └── 07_Final_Evaluation.ipynb
├── src/
│   ├── config/                  # Environment and runtime configuration
│   ├── flask_app/               # Flask application modules
│   ├── services/                # Prediction, weather, and data services
│   ├── streamlit_app/           # Streamlit dashboard application
│   └── utils/                   # Utility helpers (units, exports, formatting)
├── tests/                       # Unit/integration tests
├── .env.example                 # Environment variable template
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Installation & Setup

### 1) Clone Repository
```bash
git clone https://github.com/sachinu25/Crop_yield_production.git
cd Crop_yield_production
```

### 2) Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows
```

### 3) Install Requirements
```bash
pip install -r requirements.txt
```

### 4) Configure Environment
```bash
cp .env.example .env
# Update values as required
```

### 5) Run Applications

#### Streamlit (Recommended)
```bash
streamlit run src/streamlit_app/app.py
```

#### Flask
```bash
python -m src.flask_app.app
```

### 6) Deployment Notes
- Streamlit Cloud / containerized runtime for dashboard deployment
- Gunicorn + reverse proxy for Flask production serving
- Add CI/CD and secrets management for enterprise deployment

---

## 🖼️ Screenshots

Add screenshots in the following paths:

- `assets/screenshots/home.png`
- `assets/screenshots/prediction.png`
- `assets/screenshots/analytics.png`
- `assets/screenshots/weather.png`
- `assets/screenshots/history.png`
- `assets/screenshots/feature_importance.png`

Example markdown block:

```md
![Home](assets/screenshots/home.png)
![Prediction](assets/screenshots/prediction.png)
![Analytics](assets/screenshots/analytics.png)
![Weather](assets/screenshots/weather.png)
![History](assets/screenshots/history.png)
![Feature Importance](assets/screenshots/feature_importance.png)
```

---

## 🔭 Future Improvements

- Production-grade model monitoring and drift alerts
- Region-specific calibration layers
- Scenario simulation with forecast weather blends
- Enhanced recommendation engine with agronomy heuristics
- Dockerized one-click deployment and CI/CD quality gates

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m "feat: add amazing feature"`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

Licensed under the **MIT License**.

---

## 👤 Author

- **Author Name**
- **GitHub:** [GitHub Profile](https://github.com/your-username)
- **LinkedIn:** [LinkedIn Profile](https://www.linkedin.com/in/your-profile)
- **Email:** your.email@example.com

---

<div align="center">
  <sub>Built with data, design, and machine learning for real-world agricultural intelligence.</sub>
</div>
