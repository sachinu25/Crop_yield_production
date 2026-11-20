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
| Machine Learning | LightGBM (Optuna-tuned, production), XGBoost & CatBoost (benchmarks), native TreeSHAP |
| Visualization | Plotly, Streamlit chart components |
| Data Handling | Pandas, NumPy, Parquet |
| Configuration | python-dotenv |
| Deployment (App Runtime) | Gunicorn-compatible Flask app, Streamlit runtime |

---

## 🤖 Machine Learning Pipeline

1. **Dataset:** 249K+ APY records (2001–2020), 100+ crops, 600+ districts, enriched with Open-Meteo historical weather.
2. **Leak-free features (60):** lagged yields (expanding / 3-year rolling / EWMA, shifted by one year), weather anomalies, out-of-fold target encodings, climate PCA, and KMeans geo-clusters. Everything is fitted on training years only.
3. **Strict chronological split:** train 2001–2017, test 2018–2020. A separate dev split (2001–2014 / 2015–2017) is used for tuning.
4. **Training-derived quality filter:** test yields above 2× a crop's 2001–2017 maximum are flagged as data errors. Thresholds are saved in `models/quality_thresholds.json`.
5. **Model:** Optuna-tuned LightGBM on log1p(yield) with a per-crop smearing (bias) correction. It is benchmarked against a naive baseline, CatBoost and XGBoost.
6. **Artifacts** (loaded by the app, no retraining on startup): `models/lightgbm_model.txt`, `lightgbm_artifacts.json` (feature order, category levels, smearing factors), `quality_thresholds.json`, `evaluation.json`.
7. **Explainability:** a per-prediction SHAP waterfall from LightGBM's native TreeSHAP.

Retrain and regenerate all artifacts (about 10 minutes on a laptop CPU):
```bash
pip install -r requirements-train.txt
python scripts/train_lightgbm.py --trials 60
```

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

All numbers are **out-of-time**: models train on 2001–2017 and are scored on 2018–2020.
Tuning used a separate dev split (train 2001–2014, validate 2015–2017), and the test years were scored once.
They are generated by `scripts/train_lightgbm.py` into `models/evaluation.json` and shown on the app's **📈 Model Performance** page.

| Model | R² (all 35,614 rows) | R² (quality-filtered, 35,479 rows) | MAE (filtered) | RMSE (filtered) |
|---|---|---|---|---|
| Naive baseline (last-3-year district+crop mean) | 0.580 | 0.864 | 1.18 | 4.86 |
| CatBoost (original model) | 0.605 | 0.900 | 1.06 | 4.16 |
| XGBoost (tuned params) | 0.608 | 0.907 | 1.02 | 4.01 |
| **LightGBM, Optuna, 60 trials (production)** | **0.602** | **0.900** | **1.07** | **4.16** |

The production LightGBM cuts MAE by **9.2%** against the naive baseline on quality-filtered data.
XGBoost scores slightly higher, but LightGBM is used in the app because its model file is small (1.3 MB vs 25 MB for CatBoost) and it has native SHAP.
The earlier exploration in [`experiments/`](experiments/RESULTS.md) found the same pattern.

### Error analysis: why the full-test R² is ~0.60
- **135 test rows (0.38%)** report yields above **2× the highest yield ever recorded for that crop in 17 training years**.
  Example: guar seed at 485 t/ha, when its training maximum is 3.6 t/ha. These are almost certainly data-entry or unit errors.
- Those 135 rows cause **83% of the total squared error**. Every model (and the naive baseline) is capped near R² 0.60 on the unfiltered test set.
- The quality filter removes them using a threshold computed **from training data only**. On the remaining records, the models reach **R² 0.90–0.91**.

### Leakage checks
- Lag features use only previous years (one-year-ahead forecasting).
- Target encodings, KMeans geo-clusters and the climate PCA are fitted on training years only.

### Metric interpretation
- **R²:** share of yield variance explained by the model.
- **MAE:** average absolute error in yield units. It is easier to interpret and robust to the outliers described above.
- **RMSE:** penalises large errors heavily, so it is dominated by the few erroneous records on the full test set.

---

## 🗂️ Project Structure

```text
Crop_yield_production/
├── checkpoints/                 # Encoded datasets, train/test split artifacts
├── data/
│   ├── raw/                     # Original APY and source datasets
│   └── processed/               # Clean, model-consumable datasets
├── docs/                        # Supporting documentation and references
├── models/                      # LightGBM model + artifacts, quality thresholds, evaluation.json
├── scripts/                     # train_lightgbm.py (training + benchmark); archive/ = old experiments
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
├── requirements.txt             # Pinned runtime dependencies (Streamlit Cloud)
├── requirements-train.txt       # + Optuna, XGBoost, pytest for retraining
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

- **Author Name** Sachin Upadhyay
- **GitHub:** [GitHub Profile](https://github.com/sachinu25/)
- **LinkedIn:** [LinkedIn Profile](https://www.linkedin.com/in/sachinu25/)


---

<div align="center">
  <sub>Built with data, design, and machine learning for real-world agricultural intelligence.</sub>
</div>
