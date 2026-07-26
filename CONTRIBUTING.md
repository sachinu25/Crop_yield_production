# Contributing Guidelines

Thank you for your interest in contributing to the Indian Agriculture Crop Yield Prediction project!

## 1. Local Environment Setup

1. Clone the repository.
2. Ensure you have Python 3.10+ installed.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install test dependencies:
   ```bash
   pip install pytest pytest-cov
   ```

## 2. Testing Standards
- All new service functions must include a corresponding `pytest` function in `tests/`.
- Ensure a minimum of 80% line coverage for the `src/` directory.
- Run tests locally before submitting a PR:
  ```bash
  python -m pytest --cov=src
  ```

## 3. Code Style & Typing
- **Type Hints**: All functions must have strict PEP 484 type hints in their signature.
- **Docstrings**: Use **Google Style** docstrings for all classes, methods, and modules.
- **Logging**: Do not use `print()`. Use the standard `logging` library with appropriate log levels.
- **Exceptions**: Use custom exceptions from `src.utils.exceptions` (e.g., `DatabaseError`, `PredictionError`).

## 4. Pull Request Process
1. Create a feature branch (`feature/your-feature-name`).
2. Include a summary of your changes in the PR description.
3. Ensure the PR passes all tests and does not break the Streamlit or Flask applications.
4. Request a review from the maintainer.
