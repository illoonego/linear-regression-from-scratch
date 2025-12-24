"""Polynomial Regression implementation from scratch.

This module contains the PolynomialRegression class that implements
polynomial regression by extending linear regression with polynomial features.
"""

import warnings
from itertools import combinations_with_replacement

import numpy as np

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
            learning_rate=learning_rate, n_iterations=n_iterations, fit_intercept=fit_intercept, verbose=verbose
        )

        # Initialize state
        self.is_fitted_ = False
        self.n_features_ = None

    def fit(self, X, y, method="gradient_descent"):
        """Fit the polynomial regression model to training data.

        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features)
            y (np.ndarray): Training targets of shape (n_samples,)
            method (str): Method to use ('gradient_descent' or 'normal_equation')

        Returns:
            self: Returns self for method chaining
        """
        # === INPUT VALIDATION ===
        # Ensure X and y are numy arrays
        X = np.array(X)
        y = np.array(y)

        # Check dimensions
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must be equal")

        # Store the number of original features for later validation
        self.n_features_ = X.shape[1]

        # === TRANSFORM FEATURES ===
        X_poly = self._create_polynomial_features(X)

        # === FIT LINEAR MODEL ===
        self.linear_model_.fit(X_poly, y, method=method)

        # Mark the model as fitted
        self.is_fitted_ = True

        return self

    def predict(self, X):
        """Make predictions using the trained model.

        Args:
            X (np.ndarray): Features to predict on of shape (n_samples, n_features)

        Returns:
            np.ndarray: Predictions of shape (n_samples,)
        """

        # ===== INPUT VALIDATION =====
        # Checjk if model is fitted
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        # Convert X to numpy array
        X = np.array(X)

        # Check dimensions
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got {X.ndim}D array instead")

        # Check that X is the same number of features as training data
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Input features have {X.shape[1]} columns, but model was trained with {self.n_features_} features.")

        # === TRANSFORM FEATURES ===
        X_poly = self._create_polynomial_features(X)

        # ===== PREDICTION LOGIC =====
        # Return predictions from underlying linear model
        return self.linear_model_.predict(X_poly)

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
