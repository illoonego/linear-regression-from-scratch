"""Polynomial Regression implementation from scratch.

This module contains the PolynomialRegression class that implements
polynomial regression by extending linear regression with polynomial features.
"""

import warnings
import numpy as np
from itertools import combinations_with_replacement

from .linear_regression import LinearRegression

class PolynomialRegression:
    """Polynomial Regression implementation from scratch.

    This class implements polynomial regression by transforming features
    into polynomial features and then applying linear regression.

    Attributes:
        degree (int): Degree of polynomial features
        linear_model (LinearRegression): Underlying linear regression model

    Example:
        >>> from linear_regression.models.polynomial_regression import PolynomialRegression
        >>> import numpy as np
        >>>
        >>> # Create sample data
        >>> X = np.array([[1], [2], [3], [4], [5]])
        >>> y = np.array([1, 4, 9, 16, 25])  # y = x^2
        >>>
        >>> # Create and train model
        >>> model = PolynomialRegression(degree=2)
        >>> model.fit(X, y)
        >>>
        >>> # Make predictions
        >>> predictions = model.predict(X)
    """

    def __init__(self, degree=2, learning_rate=0.01, n_iterations=1000, fit_intercept=True, verbose=True):
        """Initialize PolynomialRegression model.

        Args:
            degree (int): Degree of polynomial features
            learning_rate (float): Learning rate for gradient descent
            n_iterations (int): Number of iterations for gradient descent
            fit_intercept (bool): Whether to fit intercept term
            verbose (bool): Whether to print progress during training
        """
        # ===== INPUT VALIDATION =====
        # Basic validation
        if not isinstance(degree, int):
            raise TypeError("degree must be an integer value")
        if degree < 1:
            raise ValueError("degree must be at least 1")

        # Warning validation
        if degree > 10:
            warnings.warn("Using a high degree may lead to overfitting")
        
        # ===== INITIALIZATION =====
        # Store hyperparameters
        self.degree = degree

        # Initialize underlying linear regression model
        self.linear_model_ = LinearRegression(
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            fit_intercept=fit_intercept,
            verbose=verbose
        )

        # Initialize state
        self.is_fitted_ = False
        self.n_features_ = None

    def fit(self, X, y, method="gradient_descent"):  # pragma: no cover
        """Fit the polynomial regression model to training data.

        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features)
            y (np.ndarray): Training targets of shape (n_samples,)
            method (str): Method to use ('gradient_descent' or 'normal_equation')

        Returns:
            self: Returns self for method chaining
        """
        # TODO: Implement fitting logic with polynomial features
        pass

    def predict(self, X):  # pragma: no cover
        """Make predictions using the trained model.

        Args:
            X (np.ndarray): Features to predict on of shape (n_samples, n_features)

        Returns:
            np.ndarray: Predictions of shape (n_samples,)
        """
        # TODO: Implement prediction logic
        pass

    def score(self, X, y):  # pragma: no cover
        """Calculate R² score (coefficient of determination).

        Args:
            X (np.ndarray): Features of shape (n_samples, n_features)
            y (np.ndarray): True targets of shape (n_samples,)

        Returns:
            float: R² score
        """
        # TODO: Implement R² calculation
        pass

    def _create_polynomial_features(self, X):
        """Transform features into polynomial features (with cross-terms).

        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features)

        Returns:
            np.ndarray: Polynomial features of shape (n_samples, n_poly_features)
        """

        # Ensure X is a numpy array (robust to list input)
        X = np.array(X)

        # List to collect all new polynomial features
        X_poly = []

        # For each degree from 1 up to self.degree (inclusive)
        for d in range(1, self.degree + 1):
            # Generate all combinations of feature indices (with replacement)
            # Each combination represents a monomial (e.g., (0, 1) -> x0*x1)
            for items in combinations_with_replacement(range(X.shape[1]), d):
                # For each sample, multiply the selected columns together
                # Example: items = (0, 1, 1) means x0 * x1^2
                new_feature = np.prod(X[:, items], axis=1).reshape(-1, 1)
                X_poly.append(new_feature)

        # Stack all new features horizontally to form the final feature matrix
        # The result shape is (n_samples, n_poly_features)
        return np.hstack(X_poly)