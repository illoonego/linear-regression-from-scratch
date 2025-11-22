"""Linear Regression from Scratch Package.

A clean, educational implementation of linear regression algorithms.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .preprocessing import StandartScaler
from .utils import train_test_split
from .metrics import mean_squared_error, r2_score
from .models.linear_regression import LinearRegression
from .models.polynomial_regression import PolynomialRegression

__all__ = [
    "LinearRegression",
    "PolynomialRegression",
    "StandartScaler",
    "train_test_split",
    "mean_squared_error",
    "r2_score"
]
