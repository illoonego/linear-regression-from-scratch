"""Linear Regression implementation from scratch.

This module contains the LinearRegression class that implements
simple and multiple linear regression using gradient descent
and normal equation methods.
"""

import warnings

import numpy as np


class LinearRegression:
    """Linear Regression implementation from scratch.

    This class implements linear regression using both gradient descent
    and normal equation methods for parameter estimation.

    Attributes:
        weights (np.ndarray): Model parameters (coefficients and intercept)
        learning_rate (float): Learning rate for gradient descent
        n_iterations (int): Number of iterations for gradient descent
        fit_intercept (bool): Whether to fit intercept term
        cost_history (list): History of cost function values during training

    Example:
        >>> from linear_regression.models.linear_regression import LinearRegression
        >>> import numpy as np
        >>>
        >>> # Create sample data
        >>> X = np.array([[1], [2], [3], [4], [5]])
        >>> y = np.array([2, 4, 6, 8, 10])
        >>>
        >>> # Create and train model
        >>> model = LinearRegression()
        >>> model.fit(X, y)
        >>>
        >>> # Make predictions
        >>> predictions = model.predict(X)
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, fit_intercept=True, verbose=True):
        """Initialize LinearRegression model.

        Args:
            learning_rate (float): Learning rate for gradient descent
            n_iterations (int): Number of iterations for gradient descent
            fit_intercept (bool): Whether to fit intercept term
            verbose (bool): Whether to print progress (cost function values) during training (default: True)
        """
        # ===== INPUT VALIDATION =====
        # Basic validation
        if learning_rate <= 0:
            raise ValueError("learning rate must be positive")
        if n_iterations <= 0:
            raise ValueError("number of iterations must be positive")
        if not isinstance(n_iterations, int):
            raise TypeError("n_iterations must be an integer value")
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be a boolean value")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean value")

        # Warning validation
        if learning_rate > 1.0:
            warnings.warn("Large learning_rate may cause convergence issues")
        if n_iterations < 10:
            warnings.warn("Very few iterations may not converge")

        # ===== INITIALIZATION =====
        # Store hyperparameters
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.verbose = verbose

        #  Initialize attributes that will be set during training
        self.weights_ = None
        self.cost_history_ = []
        self.is_fitted_ = False
        self.n_features_ = None
        self.fit_method_ = None  # Store which method was used to fit

    def fit(self, X, y, method="gradient_descent"):
        """Fit the linear regression model to training data.

        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features)
            y (np.ndarray): Training targets of shape (n_samples,)
            method (str): Method to use ('gradient_descent' or 'normal_equation')

        Returns:
            self: Returns self for method chaining
        """

        # ===== INPUT VALIDATION =====
        # Convert X and y to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Check dimensions
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got {X.ndim}D array instead")
        if y.ndim != 1:
            raise ValueError(f"y must be a 1D array, got {y.ndim}D array instead")

        # Check number of samples
        n_samples, n_features = X.shape
        if y.shape[0] != n_samples:
            raise ValueError(f"Number of samples in X and y must be equal." f"X: {n_samples}, y: {y.shape[0]}")

        # Check minimum data requirements
        if n_samples < 2:
            raise ValueError(f"At least 2 data points are required to fit the model, got {n_samples}")

        if n_features < 1:
            raise ValueError(f"At least 1 feature is required to fit the model, got {n_features}")

        # Check for non-numeric data
        if not np.isfinite(X).all():
            raise ValueError("X contains non-numeric or infinite values")
        if not np.isfinite(y).all():
            raise ValueError("y contains non-numeric or infinite values")

        # Check sufficient data points for number of features
        if n_samples < n_features + 1:
            warnings.warn(
                f"Number of samples ({n_samples}) should be greater than " f"number of features ({n_features}) for reliable results"
            )

        # Validate method parameter
        if method not in ["gradient_descent", "normal_equation"]:
            raise ValueError("method must be 'gradient_descent' or 'normal_equation'")

        # ===== FITTING LOGIC =====
        # Store number of features
        self.n_features_ = n_features
        self.fit_method_ = method  # Remember which method was used

        # Add intercept if needed
        if self.fit_intercept:
            X = self._add_intercept(X)

        # Reset cost history for new training
        self.cost_history_ = []

        # Choose fitting method
        if method == "gradient_descent":
            # Initialize initial weights only for GD
            self.weights_ = np.random.uniform(-0.1, 0.1, X.shape[1])
            self._gradient_descent(X, y)
        elif method == "normal_equation":
            # Do not initialize weights for normal equation
            self._normal_equation(X, y)

        # Mark model as fitted and return self
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
        # Check if model is fitted
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        # Convert X to numpy array
        X = np.array(X)

        # Check dimensions
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got {X.ndim}D array instead")

        # Check for non-numeric data
        try:
            if not np.isfinite(X).all():
                raise ValueError("X contains non-numeric or infinite values")
        except TypeError:
            raise ValueError("X contains non-numeric or infinite values")

        # Check that X is the same number of features as training data
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Input features have {X.shape[1]} columns, but model was trained with {self.n_features_} features.")

        # Add intercept if needed
        if self.fit_intercept:
            X = self._add_intercept(X)

        # ===== PREDICTION LOGIC =====
        # Return predictions
        return X.dot(self.weights_)

    def _add_intercept(self, X):
        """Add intercept term to feature matrix.

        Args:
            X (np.ndarray): Feature matrix

        Returns:
            np.ndarray: Feature matrix with intercept column
        """
        return np.column_stack([np.ones(X.shape[0]), X])

    def _gradient_descent(self, X, y):
        """Perform gradient descent optimization.

        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
        """

        for i in range(self.n_iterations):
            # Compute hypothesis
            h = X.dot(self.weights_)

            # Gradient of the cost function
            gradient = (1 / X.shape[0]) * X.T.dot(h - y)  # This is calculated derivative of the cost function with respect to weights

            # Update weights
            self.weights_ -= self.learning_rate * gradient

            # Compute cost for monitoring
            cost = (1 / (2 * X.shape[0])) * np.sum((h - y) ** 2)
            self.cost_history_.append(cost)

            # Print cost every 100 iterations if verbose
            if self.verbose and i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")

        return self

    def _normal_equation(self, X, y):
        """Solve using normal equation (closed-form solution).

        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
        """

        XT_X = X.T.dot(X)
        XT_y = X.T.dot(y)
        if np.linalg.matrix_rank(XT_X) == XT_X.shape[0]:
            # Non-singular: use solve
            self.weights_ = np.linalg.solve(XT_X, XT_y)
        else:
            warnings.warn("X^T X is singular or nearly singular; using pseudo-inverse for normal equation.")
            self.weights_ = np.linalg.pinv(XT_X).dot(XT_y)

        return self
