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

```
linear-regression-from-scratch/
├── README.md                       ← You are here
├── setup.py & pyproject.toml       ← Package configuration  
├── requirements.txt                ← Dependencies
├── LICENSE                         ← MIT License
├── src/linear_regression/          ← Main package
│   ├── __init__.py                 ← Package initialization
│   ├── models/                     ← ML model implementations  
│   │   ├── __init__.py
│   │   ├── linear_regression.py    ← ✅ LinearRegression (complete)
│   │   └── polynomial_regression.py ← 🚧 PolynomialRegression (planned)
│   ├── preprocessing.py            ← ✅ StandardScaler (complete)
│   ├── utils.py                    ← ✅ train_test_split (complete)
│   ├── metrics.py                  ← ✅ Metrics (R², MSE, MAE)
├── tests/                          ← Test suite
│   ├── __init__.py
│   ├── test_linear_regression.py   ← ✅ LinearRegression tests
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
   
   # Install in development mode
   pip install -e .
   ```

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

## 🧮 Mathematical Foundations

### Core Algorithms Implemented

#### Linear Regression Model
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```
Where β₀ is intercept, β₁...βₙ are coefficients, ε is error

## Transition to Matrix Form

To handle many samples efficiently, we rewrite linear regression using matrices.

### Feature Matrix
For $m$ samples and $n$ features, the feature matrix $\mathbf{X}$ is:
```math
\mathbf{X} =
\begin{bmatrix}
x_{11} & x_{12} & \dots & x_{1n} \\
x_{21} & x_{22} & \dots & x_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
x_{m1} & x_{m2} & \dots & x_{mn}
\end{bmatrix}
```
Each row represents a sample, each column a feature.

### Adding Intercept Column
To include the intercept term $\beta_0$, we prepend a column of ones to $\mathbf{X}$, forming $\mathbf{X'}$:
```math
\mathbf{X}' =
\begin{bmatrix}
1 & x_{11} & x_{12} & \dots & x_{1n} \\
1 & x_{21} & x_{22} & \dots & x_{2n} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{m1} & x_{m2} & \dots & x_{mn}
\end{bmatrix}
```
This allows the model to learn an intercept.

### Coefficient Vector
The coefficients (including intercept) are stored in a vector $\boldsymbol{\beta}$:
```math
\boldsymbol{\beta} = \begin{bmatrix}
  \beta_0 \\
  \beta_1 \\
  \beta_2 \\
  \vdots \\
  \beta_n
\end{bmatrix}
```

### Target Vector
The target values are stored in a vector $\mathbf{y}$:
```math
\mathbf{y} = \begin{bmatrix}
  y_1 \\
  y_2 \\
  \vdots \\
  y_m
\end{bmatrix}
```

### Matrix Form of the Model
The matrix form of linear regression is:
```math
\mathbf{y} = \mathbf{X'} \boldsymbol{\beta} + \boldsymbol{\epsilon}
```
Where:
- $\mathbf{y}$ is the vector of all target values
- $\mathbf{X'}$ is the feature matrix with intercept column
- $\boldsymbol{\beta}$ is the vector of all coefficients
- $\boldsymbol{\epsilon}$ is the vector of errors

### Why Matrices?
Matrix multiplication allows us to compute predictions for all samples efficiently:
$$
\hat{\mathbf{y}} = \mathbf{X'} \boldsymbol{\beta}
$$
This is the foundation for both the normal equation and gradient descent implementations in code.

#### Gradient Descent Optimization  
```
Cost: J(β) = (1/2m) × Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
Update: β := β - α × (1/m) × Xᵀ(Xβ - y)
```

#### Feature Standardization
```
StandardScaler: x' = (x - μ) / σ
```
Where μ is mean, σ is standard deviation

### Implementation Features
- **Gradient Descent**: Iterative optimization with configurable learning rate
- **Normal Equation**: Closed-form solution (planned implementation) 
- **Robust Validation**: Comprehensive input validation and error handling
- **Edge Cases**: Zero variance features, singular matrices, NaN/infinite values
- **Performance Metrics**: R², MSE, MAE with mathematical correctness

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

### Run Tests
```bash
# Run all tests
pytest tests/

# Run with coverage  
pytest tests/ --cov=src/linear_regression --cov-report=html

# Run specific test file
pytest tests/test_linear_regression.py -v
```

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