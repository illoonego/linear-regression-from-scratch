
# Development Guide

This document provides all the information needed for smooth development and collaboration on the linear-regression-from-scratch project.

## Project Setup

### Environment
- Python 3.8+
- All dependencies managed via `pyproject.toml` (PEP 621)
- No `requirements.txt` or `setup.py` required

### Installation
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# Install package and development dependencies
pip install -e .
pip install -e ".[dev,notebooks,docs]"
```

## Development Workflow

- **Formatting:**
  - Run `black .` to auto-format code
  - Run `isort .` to sort imports
- **Linting:**
  - Run `ruff .` for fast linting and code style checks
- **Testing:**
  - Run `pytest` for all tests
  - Run `pytest --cov=src/linear_regression --cov-report=html` for coverage
- **Continuous Integration:**
  - All pushes and pull requests are checked by GitHub Actions (tests, lint, coverage)

## Project Structure
```
linear-regression-from-scratch/
├── README.md                       ← You are here
├── pyproject.toml                  ← Dependencies & package configuration (PEP 621)
├── LICENSE                         ← MIT License
├── src/linear_regression/          ← Main package
│   ├── __init__.py                 ← Package initialization
│   ├── models/                     ← ML model implementations  
│   │   ├── __init__.py
│   │   ├── linear_regression.py    ← ✅ LinearRegression (gradient descent & normal equation)
│   │   └── polynomial_regression.py ← 🚧 PolynomialRegression (planned)
│   ├── preprocessing.py            ← ✅ StandardScaler (complete)
│   ├── utils.py                    ← ✅ train_test_split (complete)
│   └── metrics.py                  ← ✅ Metrics (R², MSE, MAE)
├── tests/                          ← Test suite
│   ├── __init__.py
│   ├── conftest.py                 ← ✅ Shared pytest fixtures for all tests
│   ├── test_linear_regression.py   ← ✅ LinearRegression tests (uses shared fixtures)
│   ├── test_metrics.py             ← ✅ Metrics tests (uses shared fixtures)
│   └── test_polynomial_regression.py ← 🚧 Polynomial tests (planned)
├── examples/                       ← Working examples & demos
│   ├── basic_linear_regression.py  ← ✅ Complete examples
│   ├── polynomial_regression_example.py ← 🚧 Planned
│   └── data/                       ← Sample datasets
├── notebooks/                      ← 🚧 Jupyter tutorials (planned)
├── docs/                           ← Documentation
│   ├── mathematical_background.md  ← Theory and equations
│   └── api_reference.md            ← API documentation
└── DEVELOPMENT.md                  ← Development workflow
```

**Legend**: ✅ Complete | 🚧 Planned/In Progress

## Implementation Checklist

### Core Features
- [x] `train_test_split` utility
- [x] `StandartScaler` for feature scaling
- [x] LinearRegression (gradient descent)
- [x] LinearRegression (normal equation)
- [ ] PolynomialRegression (feature transformation, integration)
- [x] Comprehensive test suite
- [x] Visual examples in notebooks
- [x] Documentation and README
## Mathematical Background

For theory and equations, see [docs/mathematical_background.md](docs/mathematical_background.md).

### Advanced Features (Optional)
- [ ] Additional optimizers (SGD, Mini-batch GD)
- [ ] Regularization (Ridge, Lasso)
- [ ] Cross-validation
- [ ] Plotting utilities
- [ ] Real dataset examples

## Collaboration & Contribution

- All contributors should follow code style enforced by Black, Ruff, and isort
- All code must pass tests and linting before merging
- Use feature branches and submit pull requests for review
- Issues and TODOs are tracked in GitHub Issues and the project TODO list
- For contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md)
- The LinearRegression class now supports a `verbose` flag for conditional training output and robust input validation for all public methods.

## Notebooks & Visual Examples
- Add new Jupyter notebooks to the `notebooks/` folder
- Use notebooks for tutorials, visualizations, and advanced examples
- Visualize regression, cost function, and optimization steps where possible

## Tips for New Contributors
- Read the README.md and docs for project overview
- Check the TODO list for open tasks
- Ask questions via GitHub Issues or Discussions
- Use clear commit messages and descriptive PR titles

## Next Steps
- Complete normal equation implementation
- Implement PolynomialRegression
- Expand test coverage
- Add more visual examples in notebooks
- Update documentation as features are added