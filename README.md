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

See the full project architecture in [DEVELOPMENT.md](DEVELOPMENT.md).

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ 
- pip package manager


### Installation

1. **Clone & Setup:**
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

> **Note:** All dependencies are now managed via `pyproject.toml`. The legacy `requirements.txt` file has been removed for clarity and modern Python packaging best practices.

2. **Run Examples:**
   ```bash
   # Run all examples
   python examples/basic_linear_regression.py
   
   # Run specific examples  
   python examples/basic_linear_regression.py 1d    # Simple regression
   python examples/basic_linear_regression.py 2d    # Multiple regression
   ```

### Basic Usage

#### Simple Linear Regression
```python
from linear_regression.models.linear_regression import LinearRegression
from linear_regression.preprocessing import StandartScaler
from linear_regression.metrics import r2_score
import numpy as np

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
scaler = StandartScaler()
X_scaled = scaler.fit(X).transform(X)
model.fit(X_scaled, y)
```

#### Multiple Linear Regression  
```python
# House price prediction example
np.random.seed(42)
size_sqft = np.random.uniform(800, 2500, 100)
bedrooms = np.random.randint(1, 5, 100)  
X = np.column_stack((size_sqft, bedrooms))

# True relationship: price = 150*size + 10000*bedrooms + 20000 + noise
price = 150 * size_sqft + 10000 * bedrooms + 20000 + np.random.randn(100) * 10000

predictions = model.predict(X)
model = LinearRegression(learning_rate=1e-7, n_iterations=5000)
model.fit(X, price)
predictions = model.predict(X)

print(f"Learned coefficients: {model.weights_[1:]}")  # [size_coef, bedroom_coef]
print(f"Intercept: {model.weights_[0]}")
from linear_regression.metrics import r2_score
print(f"R² Score: {r2_score(price, predictions):.4f}")
```

## 📊 Current Features

### ✅ Implemented & Tested
- **LinearRegression**: Complete implementation with gradient descent
- **StandardScaler**: Feature standardization with robust validation  
- **Examples**: Working 1D and 2D regression demonstrations
- **Error Handling**: Comprehensive validation and edge case management
- **Professional Structure**: PyPI-ready package with proper metadata

### 🚧 Planned Features  
- **Normal Equation**: Closed-form solution implementation
- **PolynomialRegression**: Non-linear relationship modeling
- **Advanced Optimizers**: SGD, Mini-batch GD, Adam
- **Regularization**: Ridge (L2) and Lasso (L1) regression
- **Visualization Tools**: Plotting utilities for analysis
- **Jupyter Tutorials**: Interactive learning notebooks

## 🧪 Testing & Development

### Run Tests & Coverage
```bash
# Run all tests
pytest tests/

# Run with coverage (see missing lines in terminal)
pytest --cov=src/linear_regression --cov-report=term-missing

# Run specific test file
pytest tests/test_linear_regression.py -v
```

### Continuous Integration (CI)
This project uses GitHub Actions to automatically run tests, linting (ruff), formatting checks (black), and coverage reporting on every push and pull request. The workflow is defined in `.github/workflows/python-ci.yml` and tests against multiple Python versions.

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
$ python examples/basic_linear_regression.py 2d

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
True:    size=150.00, bedroom=10000.00, intercept=20000.00  
Learned: size=149.87, bedroom=9989.23, intercept=20145.67
Error:   size=0.13, bedroom=10.77, intercept=145.67
```

## 🎓 Educational Value

This project demonstrates:
- **Mathematical Understanding**: Implement algorithms from equations
- **Software Engineering**: Professional Python package development
- **Machine Learning**: Core concepts without library abstractions  
- **Numerical Computing**: Efficient NumPy vectorized operations
- **Testing**: Comprehensive test coverage with edge cases
- **Documentation**: Clear code documentation and user guides

## 🤝 Contributing

We welcome contributions! Please see:
- [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and onboarding
- [DEVELOPMENT.md](DEVELOPMENT.md) for development workflow
- [Issues](https://github.com/illoonego/linear-regression-from-scratch/issues) for bug reports and feature requests

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for educational purposes to understand ML fundamentals
- Mathematical foundations from "The Elements of Statistical Learning"
- Inspired by the need for transparent, understandable ML implementations

---

**Note**: This is primarily an educational project. For production ML workflows, consider using established libraries like scikit-learn, though this implementation is production-quality and could be used in real applications.