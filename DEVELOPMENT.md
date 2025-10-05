# Development Commands

## Setup Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (macOS/Linux)
source venv/bin/activate

# Activate virtual environment (Windows)
# venv\Scripts\activate

# Install package in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Install all dependencies (including notebooks)
pip install -e ".[dev,notebooks,docs]"
```

## Development Workflow

```bash
# Format code
black src/ tests/ examples/

# Sort imports
isort src/ tests/ examples/

# Lint code
flake8 src/ tests/ examples/

# Run tests
pytest

# Run tests with coverage
pytest --cov=src/linear_regression --cov-report=html

# Run specific test file
pytest tests/test_linear_regression.py

# Run specific test
pytest tests/test_linear_regression.py::TestLinearRegression::test_fit
```

## Project Structure Overview

```
├── src/linear_regression/           # Main package
│   ├── __init__.py                 # Package initialization
│   ├── models/                     # Model implementations
│   │   ├── __init__.py
│   │   ├── linear_regression.py    # LinearRegression class
│   │   └── polynomial_regression.py # PolynomialRegression class
│   └── optimizers/                 # Optimization algorithms
│       └── __init__.py
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── test_linear_regression.py
│   └── test_polynomial_regression.py
├── examples/                       # Usage examples
│   ├── basic_linear_regression.py
│   ├── polynomial_regression_example.py
│   └── data/
│       └── sample_data.csv
├── notebooks/                      # Jupyter notebooks
├── docs/                          # Documentation
│   ├── mathematical_background.md
│   └── api_reference.md
├── requirements.txt               # Dependencies
├── setup.py                      # Package setup (legacy)
├── pyproject.toml                # Modern package configuration
├── README.md                     # Project documentation
├── LICENSE                       # License file
└── .gitignore                   # Git ignore rules
```

## Implementation Checklist

### Core Features
- [ ] LinearRegression class
  - [ ] Gradient descent optimization
  - [ ] Normal equation solution
  - [ ] Prediction functionality
  - [ ] Model evaluation metrics
- [ ] PolynomialRegression class
  - [ ] Polynomial feature transformation
  - [ ] Integration with LinearRegression
- [ ] Comprehensive test suite
- [ ] Documentation and examples

### Advanced Features (Optional)
- [ ] Additional optimizers (SGD, Mini-batch GD)
- [ ] Regularization (Ridge, Lasso)
- [ ] Feature scaling utilities
- [ ] Cross-validation
- [ ] Plotting utilities
- [ ] Real dataset examples

## Next Steps

1. Implement the LinearRegression class in `src/linear_regression/models/linear_regression.py`
2. Implement the PolynomialRegression class in `src/linear_regression/models/polynomial_regression.py`
3. Write comprehensive tests
4. Create working examples
5. Add Jupyter notebook tutorials
6. Expand documentation as needed