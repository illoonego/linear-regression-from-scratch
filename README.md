# Linear Regression from Scratch

A production-quality, educational implementation of linear regression algorithms built from scratch using NumPy. This library provides clean, well-documented implementations for learning the mathematical foundations of linear regression while maintaining professional-grade code quality.

## 🎯 Project Overview

This project implements linear regression algorithms **from first principles** without using high-level ML libraries like scikit-learn. It's designed as both an educational tool and a functional library that demonstrates professional Python package development practices.

### 🌟 What Makes This Special

- **📚 Educational Focus**: Understand the mathematics behind linear regression
- **🏗️ Production Quality**: Professional package structure ready for PyPI
- **🔬 From Scratch**: Only NumPy used for mathematical operations  
- **🧪 Fully Tested**: Comprehensive test suite with edge case handling
- **📦 Complete Package**: Installable via pip with proper dependency management

## 📁 Project Architecture

See the full project architecture in [CONTRIBUTING.md](https://github.com/illoonego/linear-regression-from-scratch/blob/main/CONTRIBUTING.md).

## 📐 Mathematical Background

For a detailed explanation of the mathematical foundations behind linear regression, see [mathematical_background.md](https://github.com/illoonego/linear-regression-from-scratch/blob/main/docs/mathematical_background.md).

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ 
- pip package manager


### Installation

**Option 1: Install from PyPI (recommended)**
```bash
pip install linreg-from-scratch
```

**Option 2: Clone & Setup for development**
```bash
git clone https://github.com/illoonego/linear-regression-from-scratch.git
cd linear-regression-from-scratch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies using pyproject.toml (PEP 621)
pip install -e .[dev]
# For optional dependencies (notebooks, docs):
pip install -e ".[notebooks,docs]"
```

> **Note:** All dependencies are managed via `pyproject.toml`.

### Running Examples

Linear Regression Usage Examples:

```bash
# Run all examples
python examples/linear_example.py

# Run specific examples
python examples/linear_example.py 1d    # Simple regression
python examples/linear_example.py 2d    # Multiple regression
```

Polynomial Regression Usage Examples:

```bash
# Run all examples
python examples/polynomial_example.py
```

### Basic Usage

#### Simple Linear Regression
```python
import numpy as np
from linear_regression import LinearRegression, StandardScaler, r2_score

# Create sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2.1, 3.9, 6.1, 8.0, 9.9])  # y ≈ 2x with noise

# Option 1: Direct usage
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y, method='gradient_descent')
predictions = model.predict(X)
print(f"Weights: {model.weights_}")
print(f"R² Score: {r2_score(y, predictions):.4f}")

# Option 2: With preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit(X).transform(X)
model.fit(X_scaled, y)
predictions_scaled = model.predict(X_scaled)
print(f"Weights (scaled): {model.weights_}")
print(f"R² Score (scaled): {r2_score(y, predictions_scaled):.4f}")
```

#### Multiple Linear Regression  
```python
import numpy as np
from linear_regression import LinearRegression, r2_score

# House price prediction example
np.random.seed(42)
size_sqft = np.random.uniform(800, 2500, 100)
bedrooms = np.random.randint(1, 5, 100)
X = np.column_stack((size_sqft, bedrooms))

# True relationship: price = 150*size + 10000*bedrooms + 20000 + noise
price = 150 * size_sqft + 10000 * bedrooms + 20000 + np.random.randn(100) * 10000

model = LinearRegression(learning_rate=1e-7, n_iterations=5000)
model.fit(X, price)
predictions = model.predict(X)

print(f"Learned coefficients: {model.weights_[1:]}")  # [size_coef, bedroom_coef]
print(f"Intercept: {model.weights_[0]}")
print(f"R² Score: {r2_score(price, predictions):.4f}")
```

#### Polynomial Regression Example
```python
import numpy as np
from linear_regression import PolynomialRegression, r2_score

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])  # y = x^2
model = PolynomialRegression(degree=2)
model.fit(X, y)
predictions = model.predict(X)
print(f"Predictions: {predictions}")
print(f"R² Score: {r2_score(y, predictions):.4f}")
```

## 📊 Current Features

### ✅ Implemented & Tested
- **LinearRegression**: Complete implementation with both gradient descent and normal equation (closed-form solution)
- **PolynomialRegression**: Full implementation with feature expansion, robust input validation, and edge case handling
- **StandardScaler**: Feature standardization with robust validation
- **Metrics**: r2_score, mean_squared_error, mean_absolute_error (100% coverage)
- **Examples**: Working 1D, 2D, and polynomial regression demonstrations
- **Error Handling**: Comprehensive input validation and edge case management
- **Verbose Training Output**: Control progress printing with the `verbose` flag
- **Professional Structure**: PyPI-ready package with proper metadata

### 🚧 Planned Features  
See the [CONTRIBUTING.md](https://github.com/illoonego/linear-regression-from-scratch/blob/main/CONTRIBUTING.md) for the full roadmap and planned features.

## 🛠️ For GitHub Users & Contributors

The following sections are relevant for users who clone the repository from GitHub (not for PyPI package users):

## 🧪 Testing & Development

### Run Tests & Coverage
```bash
# Run all tests and coverage
pytest --cov=src/linear_regression tests/ -v
# Coverage: LinearRegression 99%, PolynomialRegression & metrics 100%
```

### Continuous Integration & Delivery (CI/CD)
This project uses GitHub Actions for:
- **CI:** Automatic tests, linting (ruff), formatting checks (black), and coverage reporting on every push and pull request. See `.github/workflows/python-ci.yml`.
- **CD:** Automated publishing to PyPI on new version tags. See `.github/workflows/python-cd.yml`.

**How releases work:**
- When a new version tag (e.g., `v1.0.0`) is pushed, the CD workflow builds and publishes the package to PyPI using secure repository secrets.
- See [CONTRIBUTING.md](https://github.com/illoonego/linear-regression-from-scratch/blob/main/CONTRIBUTING.md) for more on the release workflow.

### Code Quality
```bash
# Format code
black src/ tests/ examples/

# Sort imports  
isort src/ tests/ examples/

# Lint code
flake8 src/ tests/ examples/
```

### Development Installation
```bash
# Install with development dependencies
pip install -e ".[dev,notebooks,docs]"
```

## 🎯 Example Output

```bash
$ python examples/linear_example.py 2d

2D Multiple Linear Regression Example
----------------------------------------

Generating synthetic data...
Data points: 100
True weights: size coefficient=150, bedroom coefficient=10000, intercept=20000

Training model with Gradient Descent...
Iteration 0: Cost = 1250000000.0000
Iteration 500: Cost = 125678923.4567  
Iteration 1000: Cost = 89234567.1234

Training completed!

Results:
Learned weights: size coefficient=149.87, bedroom coefficient=9989.23, intercept=20145.67
R² Score:        0.9234
MSE:             89234567.12

Comparison with True Values:
model_gd = LinearRegression(learning_rate=0.01, n_iterations=1000, verbose=True)
model_gd.fit(X, y, method='gradient_descent')
predictions_gd = model_gd.predict(X)
print(f"GD Weights: {model_gd.weights_}")
print(f"GD R² Score: {r2_score(y, predictions_gd):.4f}")

# Option 2: Normal Equation (closed-form)
model_ne = LinearRegression(verbose=False)
model_ne.fit(X, y, method='normal_equation')
predictions_ne = model_ne.predict(X)
print(f"NE Weights: {model_ne.weights_}")
print(f"NE R² Score: {r2_score(y, predictions_ne):.4f}")
True:    size=150.00, bedroom=10000.00, intercept=20000.00  
Learned: size=149.87, bedroom=9989.23, intercept=20145.67
Error:   size=0.13, bedroom=10.77, intercept=145.67
```

## 📏 Metrics Usage Example
```python
from linear_regression import r2_score, mean_squared_error, mean_absolute_error
print(r2_score(y, predictions))
print(mean_squared_error(y, predictions))
print(mean_absolute_error(y, predictions))
```

## 🎓 Educational Value

This project demonstrates:
- **Mathematical Understanding**: Implement algorithms from equations, including polynomial feature expansion
- **Software Engineering**: Professional Python package development
- **Machine Learning**: Core concepts without library abstractions
- **Numerical Computing**: Efficient NumPy vectorized operations
- **Testing**: Comprehensive test coverage with edge cases and fixtures
- **Documentation**: Clear code documentation and user guides

## 🤝 Contributing

We welcome contributions! Please see:
- [CONTRIBUTING.md](https://github.com/illoonego/linear-regression-from-scratch/blob/main/CONTRIBUTING.md) for guidelines and onboarding
- [Issues](https://github.com/illoonego/linear-regression-from-scratch/issues) for bug reports and feature requests

## 🙏 Acknowledgments
- Built for educational purposes to understand ML fundamentals
- Mathematical foundations from "The Elements of Statistical Learning"
- Inspired by the need for transparent, understandable ML implementations

---

**Note**: This is primarily an educational project. For production ML workflows, consider using established libraries like scikit-learn, though this implementation is production-quality and could be used in real applications.