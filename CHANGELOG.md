# CHANGELOG

## [Unreleased] - 2026-07-23

### Added
- **Machine Learning**: Implemented CatBoost algorithm with robust chronological cross-validation to prevent temporal leakage.
- **Explainability**: Integrated SHAP values to explain feature impacts on predictions.
- **Testing**: Added exhaustive `pytest` suite with mock data, mock API responses, and coverage tracking (`pytest-cov`).
- **Error Handling**: Centralized custom exceptions in `src/utils/exceptions.py` for Database, API, and Validation errors.
- **Logging**: Migrated all raw `print()` statements to Python's standard `logging` library for production readiness.
- **Reporting**: Automated error analysis generation (`error_analysis.md`) and model comparison metrics (`model_report.md`).
- **Data Validation**: Pipeline now automatically exports `validation_report.csv` identifying nulls and boundary constraint violations.

### Changed
- **Architecture**: Separated frontend Streamlit/Flask logic from backend services.
- **Database**: Migrated local SQLite pipeline to MySQL star schema for Power BI DirectQuery integration.
- **Type Hints**: Implemented strict PEP 484 type hints across all service modules.
- **Documentation**: Overhauled `README.md`, Model Card, and Data Dictionary to meet FAANG MLE portfolio standards.

### Removed
- **Legacy Files**: Safely archived unused `/agriculture_20*` and `/weather_20*` raw dataset directories to reduce repository bloat.
- **Redundant Code**: Consolidated scattered CSV loading and geocoding into `data_service.py`.
