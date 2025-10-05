# Linear Regression from Scratch

A clean, educational implementation of linear regression algorithms built from scratch using only NumPy and standard Python libraries.

## 🎯 Project Overview

This project implements linear regression algorithms without using high-level machine learning libraries like scikit-learn. It's designed for educational purposes to understand the mathematical foundations and implementation details of linear regression.

## 📁 Project Structure

```
linear-regression-from-scratch/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── .python-version
├── src/
│   └── linear_regression/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── linear_regression.py
│       │   └── polynomial_regression.py
│       └── optimizers/
│           └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── test_linear_regression.py
│   └── test_polynomial_regression.py
├── examples/
│   ├── basic_linear_regression.py
│   ├── polynomial_regression_example.py
│   └── data/
│       └── sample_data.csv
├── notebooks/
│   ├── linear_regression_tutorial.ipynb
│   └── polynomial_regression_tutorial.ipynb
└── docs/
    ├── mathematical_background.md
    └── api_reference.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/linear-regression-from-scratch.git
   cd linear-regression-from-scratch
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   # venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

### Basic Usage

```python
from linear_regression.models.linear_regression import LinearRegression
import numpy as np

# Create sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
print(f"Predictions: {predictions}")
```

## 🧮 Mathematical Background

This implementation covers:

- **Simple Linear Regression**: $y = \beta_0 + \beta_1 x + \epsilon$
- **Multiple Linear Regression**: $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon$
- **Polynomial Regression**: $y = \beta_0 + \beta_1 x + \beta_2 x^2 + ... + \beta_n x^n + \epsilon$

### Optimization Methods
- Gradient Descent
- Normal Equation (Closed-form solution)
- Stochastic Gradient Descent

## 📊 Features

- ✅ Simple Linear Regression
- ✅ Multiple Linear Regression  
- ✅ Polynomial Regression
- ✅ Gradient Descent Optimization
- ✅ Normal Equation Solution
- ✅ Model Evaluation Metrics (MSE, MAE, R²)
- ✅ Data Visualization Tools
- ✅ Comprehensive Test Suite
- ✅ Jupyter Notebook Tutorials

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src/linear_regression --cov-report=html
```

## 📚 Examples

Check out the `examples/` directory for:
- Basic linear regression example
- Polynomial regression with different degrees
- Comparison of optimization methods
- Real-world dataset examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for educational purposes to understand machine learning fundamentals
- Inspired by classic statistical learning theory
- Mathematical foundations based on "The Elements of Statistical Learning"

## 📞 Contact

- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

**Note**: This is an educational project. For production use, consider using established libraries like scikit-learn.