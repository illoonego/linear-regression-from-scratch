"""Linear Regression from Scratch Package.

A clean, educational implementation of linear regression algorithms.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .models.linear_regression import LinearRegression
from .models.polynomial_regression import PolynomialRegression

__all__ = [
    "LinearRegression",
    "PolynomialRegression",
]
