"""Linear Regression from Scratch Package.

A clean, educational implementation of linear regression algorithms.
"""

__version__ = "0.1.0"

from .metrics import mean_squared_error, r2_score
from .models.linear_regression import LinearRegression
from .models.polynomial_regression import PolynomialRegression
from .preprocessing import StandardScaler
from .utils import train_test_split

__all__ = [
    "LinearRegression",
    "PolynomialRegression",
    "StandardScaler",
    "train_test_split",
    "mean_squared_error",
    "r2_score",
]
