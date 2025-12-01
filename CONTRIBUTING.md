
# Contributing to linear-regression-from-scratch

Thank you for your interest in contributing! This guide will help you get started and ensure a smooth development experience for everyone.

---

## Getting Started Guide

1. **Clone the Repository**
  ```bash
  git clone https://github.com/illoonego/linear-regression-from-scratch.git
  cd linear-regression-from-scratch
  ```
2. **Create and Activate a Virtual Environment**
  ```bash
  python -m venv venv
  source venv/bin/activate  # macOS/Linux
  # venv\Scripts\activate  # Windows
  ```
3. **Install Dependencies**
  ```bash
  pip install -e .
  pip install -e ".[dev,notebooks,docs]"
  ```
4. **Use Top-Level Imports**
  - When writing examples, tests, or documentation, use top-level imports for public API:
    ```python
    from linear_regression import LinearRegression, StandardScaler, r2_score, mean_squared_error, train_test_split
    ```
4. **Run Tests**
  ```bash
  pytest
  ```
5. **Format, Lint, and Check Code**
  ```bash
  black .
  isort .
  ruff .
  ```
6. **Explore Examples and Notebooks**
  - See the `examples/` and `notebooks/` folders for usage and visualizations.

---

## Architecture Overview

**Main Components:**
- `src/linear_regression/`: Core package
  - `models/`: Model implementations (LinearRegression, PolynomialRegression)
  - `optimizers/`: Optimization algorithms
  - `preprocessing.py`: Feature scaling utilities
  - `utils.py`: Data splitting and helpers
  - `metrics.py`: Evaluation metrics
- `tests/`: Unit and integration tests
- `examples/`: Usage scripts and sample data
- `notebooks/`: Jupyter notebooks for tutorials and visualization
- `docs/`: Documentation (API reference, mathematical background)

**Data Flow:**
1. Data Preparation: Load and preprocess data (scaling, splitting)
2. Model Training: Fit models using gradient descent or normal equation
3. Prediction: Use trained models to predict new data
4. Evaluation: Assess model performance with metrics
5. Visualization: Use notebooks for visual analysis

---

## Contributor Code of Conduct

All contributors are expected to:
- Be respectful and inclusive in all interactions
- Provide constructive feedback and accept reviews graciously
- Avoid personal attacks, harassment, or discrimination
- Use clear, professional language in issues, PRs, and discussions
- Collaborate openly and help others when possible

Violations may result in removal from the project or reporting to platform administrators.

For questions or concerns, contact the project maintainer.

---

## How to Contribute

- **Fork the repository** and clone your fork locally.
- **Create a feature branch** for your changes (e.g., `feature/normal-equation`).
- **Make your changes** with clear code, comments, and tests.
- **Run formatting and linting** before submitting:
  - `black .`
  - `isort .`
  - `ruff .`
- **Run all tests** with `pytest` and ensure coverage is maintained.
- **Push your branch** and open a pull request (PR) with a clear description.
- **Participate in code review** and address feedback promptly.

---

## Code Style & Quality

- Use [Black](https://black.readthedocs.io/) for formatting.
- Use [Ruff](https://docs.astral.sh/ruff/) for linting.
- Use [isort](https://pycqa.github.io/isort/) for import sorting.
- Write clear, concise docstrings and comments.
- Add or update tests for all new features and bug fixes.
- Use correct naming: `StandardScaler` (not `StandartScaler`).

---

## Issue Tracking & Labels

- Use GitHub Issues to report bugs, request features, or ask questions.
- Use clear, descriptive titles and provide context.
- Use the following labels to help organize issues:
  - `bug`: For bug reports
  - `enhancement`: For feature requests
  - `question`: For general questions
  - `good first issue`: For beginner-friendly tasks
  - `help wanted`: For tasks needing extra help

**Issue Templates:**
- Bug Report: Describe the bug, steps to reproduce, expected behavior, environment, and screenshots if applicable.
- Feature Request: Describe the problem, solution, alternatives, and context.
- Question: Ask your question and provide any relevant details.

---

## Tips for Contributors

- Read the [DEVELOPMENT.md](DEVELOPMENT.md) for workflow and setup details.
- Check the TODO list and open issues for tasks.
- Ask questions via Issues or Discussions if you need help.
- Be respectful and collaborative in all interactions.

---

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
