# Contributing to linear-regression-from-scratch

This document provides all the information needed for smooth development and collaboration on the linear-regression-from-scratch project! This guide will help you get started and ensure a smooth development experience for everyone.

---

## Getting Started Guide

0. **Environment**
- Python 3.8+
- All dependencies managed via `pyproject.toml` (PEP 621)
- No `requirements.txt` or `setup.py` required

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
  from linear_regression import LinearRegression, PolynomialRegression, StandardScaler, r2_score, mean_squared_error, mean_absolute_error, train_test_split
  ```
4. **Run Tests & Coverage**
  ```bash
  pytest                                       # for all tests
  pytest --cov=src/linear_regression tests/ -v # for coverage
  ```
5. **Format, Lint, and Check Code**
  - Run `black .` to auto-format code
  - Run `isort .` to sort imports
  - Run `ruff .` for fast linting and code style checks

6. **Continuous Integration & Delivery (CI/CD):**
  - All pushes and pull requests are checked by GitHub Actions for tests, lint, and coverage (`python-ci.yml`).
  - Releases are published to PyPI automatically when a new version tag (e.g., `v1.0.0`) is pushed (`pypi-cd.yml`).
  - Test releases are published to TestPyPI automatically when a test tag (e.g., `test-v1.0.0`) is pushed (`testpypi-cd.yml`).
  - PyPI credentials are stored securely as GitHub repository secrets (`PYPI_API_TOKEN`).
  - See `.github/workflows/` for workflow files.

7. **Explore Examples and Notebooks**
  - See the `examples/` and `notebooks/` folders for usage and visualizations.

---

## Mathematical Background

For theory and equations, see [docs/mathematical_background.md](docs/mathematical_background.md).

---

## Architecture Overview

```
linear-regression-from-scratch/
├── README.md                       ← Project overview & usage
├── pyproject.toml                  ← Dependencies & package configuration (PEP 621)
├── LICENSE                         ← MIT License
├── src/
│   └── linear_regression/
│       ├── __init__.py             ← Package initialization
│       ├── models/
│       │   ├── __init__.py
│       │   ├── linear_regression.py    ← ✅ LinearRegression (complete)
│       │   └── polynomial_regression.py ← ✅ PolynomialRegression (complete)
│       ├── preprocessing.py        ← ✅ StandardScaler (complete)
│       ├── utils.py                ← ✅ train_test_split (complete)
│       └── metrics.py              ← ✅ Metrics (R², MSE, MAE)
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 ← ✅ Shared pytest fixtures for all tests
│   ├── test_linear_regression.py   ← ✅ LinearRegression tests
│   ├── test_metrics.py             ← ✅ Metrics tests
│   ├── test_polynomial_regression.py ← ✅ PolynomialRegression tests
│   ├── test_preprocessing.py       ← ✅ Preprocessing tests
│   └── test_utils.py               ← ✅ Utils tests
├── examples/                       ← 🚧 Usage Example (In Progress)
│   ├── linear_example.py           ← ✅ Linear regression example
│   ├── polynomial_example.py       ← ✅ Polynomial regression example
│   └── data/                       ← Sample datasets
├── notebooks/                      ← 🚧 Jupyter tutorials (In Progress)
├── docs/
│   ├── mathematical_background.md  ← Theory and equations
│   └── api_reference.md            ← API documentation
├── DEVELOPMENT.md                  ← Development workflow
├── CONTRIBUTING.md                 ← Contribution guidelines
└── .github/
     └── workflows/
        ├── python-ci.yml           ← CI: tests, lint, coverage
        ├── pypi-cd.yml             ← CD: PyPI publishing
        └── testpypi-cd.yml         ← CD: TestPyPI publishing
```

**Legend**: ✅ Complete | 🚧 Planned/In Progress | (optional/expandable)

---

## Implementation Checklist (TODO)

### Core Features
- [x] `train_test_split` utility
- [x] `StandardScaler` for feature scaling
- [x] LinearRegression (gradient descent & normal equation)
- [x] PolynomialRegression (feature transformation, integration)
- [x] Comprehensive test suite (all modules)
- [x] Visual examples in examples/
- [x] Documentation and README

### Roadmap & Planned Features
- [ ] Advanced optimizers (SGD, Mini-batch GD, Adam)
- [ ] Regularization (Ridge, Lasso)
- [ ] Visualization tools (plotting utilities)
- [ ] Jupyter tutorials (notebooks/)
- [ ] Real dataset examples
- [ ] Cross-validation

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
- **Push your branch** and open a pull request (PR) with a clear, descriptive summary of your changes.
- **Participate in code review** and address feedback promptly and respectfully.

---

## Code Style & Quality


- Use [Black](https://black.readthedocs.io/) for formatting.
- Use [Ruff](https://docs.astral.sh/ruff/) for linting.
- Use [isort](https://pycqa.github.io/isort/) for import sorting.
- Write clear, concise docstrings and comments.
- Add or update tests for all new features and bug fixes.
- Use correct naming: `StandardScaler` (not `StandartScaler`).

---

## Naming Conventions

- **Classes:** Use `CamelCase` (e.g., `LinearRegression`, `PolynomialRegression`, `StandardScaler`)
- **Functions/Methods:** Use `snake_case` (e.g., `fit`, `predict`, `train_test_split`)
- **Variables:** Use `snake_case` (e.g., `learning_rate`, `num_iterations`)
- **Constants:** Use `UPPER_CASE` (e.g., `DEFAULT_SEED`)
- Be descriptive but concise; avoid abbreviations unless well-known.

---

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

- Check the TODO list and open issues for tasks.
- Ask questions via Issues or Discussions if you need help.
- Be respectful and collaborative in all interactions.

---

## CI/CD Workflows

This project uses GitHub Actions for:
- **CI:** Automatic tests, linting, formatting, and coverage on every push and pull request. See `.github/workflows/python-ci.yml`.
- **CD:**
  - **PyPI:** Automated publishing to PyPI on new version tags. See `.github/workflows/pypi-cd.yml`.
  - **TestPyPI:** Automated publishing to TestPyPI on test tags (e.g., `test-v1.0.0`). See `.github/workflows/testpypi-cd.yml`.

---

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
