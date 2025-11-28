"""Tests for LinearRegression class."""

import numpy as np
import pytest

from linear_regression.models.linear_regression import LinearRegression


class TestLinearRegression:
    """Test cases for LinearRegression class."""

    def test_initialization(self):
        """Test model initialization."""
        # Test 1: Default parameters
        model = LinearRegression()

        # Check hyperparameters are stored correctly
        assert model.learning_rate == 0.01
        assert model.n_iterations == 1000
        assert model.fit_intercept is True

        # Check initial state of fitted attributes
        assert model.weights_ is None
        assert model.cost_history_ == []
        assert model.is_fitted_ is False
        assert model.n_features_ is None
        assert model.fit_method_ is None

        # Test 2: Custom parameters
        model_custom = LinearRegression(learning_rate=0.05, n_iterations=500, fit_intercept=False)

        # Check custom hyperparameters
        assert model_custom.learning_rate == 0.05
        assert model_custom.n_iterations == 500
        assert model_custom.fit_intercept is False

        # Check initial state of fitted attributes
        assert model_custom.weights_ is None
        assert model_custom.cost_history_ == []
        assert model_custom.is_fitted_ is False
        assert model_custom.n_features_ is None
        assert model_custom.fit_method_ is None

    def test_fit_gradient_descent(self, model, synthetic_data):
        """Test fitting with gradient descent."""
        X, y = synthetic_data
        model.fit(X, y, method="gradient_descent")
        assert model.is_fitted_ is True
        assert model.weights_ is not None
        assert len(model.cost_history_) == model.n_iterations
        assert model.fit_method_ == "gradient_descent"

    def test_fit_normal_equation(self, model, synthetic_data):
        """Test fitting with normal equation (non-singular and singular cases)."""
        X, y = synthetic_data
        # Non-singular case: features are independent
        model.fit(X, y, method="normal_equation")
        assert model.is_fitted_ is True
        assert model.weights_ is not None
        assert len(model.cost_history_) == 0  # No cost history for normal equation
        assert model.fit_method_ == "normal_equation"

        # Singular case: duplicate columns
        X_sing = X.copy()
        X_sing = np.column_stack([X_sing, X_sing[:, 0]])  # Add duplicate column
        model_sing = LinearRegression()

        with pytest.warns(UserWarning, match="singular or nearly singular"):
            model_sing.fit(X_sing, y, method="normal_equation")
        assert model_sing.is_fitted_ is True
        assert model_sing.weights_ is not None
        assert len(model_sing.cost_history_) == 0
        assert model_sing.fit_method_ == "normal_equation"

    def test_predict(self, model, synthetic_data):
        """Test prediction functionality."""
        X, y = synthetic_data
        model.fit(X, y, method="gradient_descent")
        predictions = model.predict(X)
        assert predictions.shape == y.shape

        # Test: Predict before fitting
        model_unfit = LinearRegression()
        with pytest.raises(ValueError, match="Model has not been fitted yet"):
            model_unfit.predict(X)

        # Test: 1D input (should raise ValueError)
        X_1d = X[:, 0]
        with pytest.raises(ValueError, match="X must be a 2D array"):
            model.predict(X_1d)

        # Test: Wrong feature count
        # Always pass input with a different number of columns than training data
        n_samples, n_features = X.shape
        # If training on 1 feature, test with 2; else test with 1
        if n_features == 1:
            X_wrong = np.hstack([X, X])  # shape (n_samples, 2)
        else:
            X_wrong = X[:, :1]  # shape (n_samples, 1)
        with pytest.raises(ValueError, match="Input features have"):
            model.predict(X_wrong)

        # Test: Non-numeric input
        X_bad = X.astype(object)
        X_bad[0, 0] = None
        with pytest.raises(ValueError, match="non-numeric or infinite"):
            model.predict(X_bad)

    def test_add_intercept(self, model):
        """Test intercept addition."""
        # TODO: Test intercept functionality
        pass

    def test_simple_linear_regression(self):
        """Test simple linear regression on known data."""
        # TODO: Test on y = 2x + 1 type data
        pass

    def test_multiple_linear_regression(self):
        """Test multiple linear regression."""
        # TODO: Test on multiple features
        pass

    def test_edge_cases(self):
        """Test edge cases and error handling."""

        # Test invalid learning rates
        with pytest.raises(ValueError, match="learning rate must be positive"):
            LinearRegression(learning_rate=0)

        with pytest.raises(ValueError, match="learning rate must be positive"):
            LinearRegression(learning_rate=-0.01)

        # Test invalid iterations
        with pytest.raises(ValueError, match="number of iterations must be positive"):
            LinearRegression(n_iterations=0)

        with pytest.raises(TypeError, match="n_iterations must be an integer value"):
            LinearRegression(n_iterations=1000.5)

        # Test fit_intercept type
        with pytest.raises(TypeError, match="fit_intercept must be a boolean value"):
            LinearRegression(fit_intercept="yes")

    # Test for verbose flag
    def test_verbose_flag(self, synthetic_data, capsys):
        """Test that verbose flag controls printing during gradient descent."""
        X, y = synthetic_data
        # Use small n_iterations for test speed
        model_verbose = LinearRegression(learning_rate=0.01, n_iterations=101, verbose=True)
        model_verbose.fit(X, y, method="gradient_descent")
        out = capsys.readouterr().out
        assert "Iteration 0: Cost" in out
        assert "Iteration 100: Cost" in out

        model_silent = LinearRegression(learning_rate=0.01, n_iterations=101, verbose=False)
        model_silent.fit(X, y, method="gradient_descent")
        out_silent = capsys.readouterr().out
        assert "Iteration" not in out_silent
