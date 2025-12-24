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

    def test_fit_normal_equation_edge_cases(self):
        import numpy as np

        from linear_regression.models.linear_regression import LinearRegression

        # Singular matrix: duplicate columns
        X = np.array([[1, 2], [2, 4], [3, 6]])
        y = np.array([1, 2, 3])
        model = LinearRegression()
        with pytest.warns(UserWarning, match="singular or nearly singular"):
            model.fit(X, y, method="normal_equation")
        assert model.is_fitted_ is True
        assert model.weights_ is not None
        # Small sample size warning
        X_small = np.array([[1, 2, 3], [4, 5, 6]])
        y_small = np.array([1, 2])
        model2 = LinearRegression()
        with pytest.warns(UserWarning, match=r"Number of samples \(2\) should be greater than number of features \(3\)"):
            model2.fit(X_small, y_small, method="normal_equation")
        # Non-numeric X
        X_bad = np.array([[1, np.nan], [2, 3]])
        y_bad = np.array([1, 2])
        model3 = LinearRegression()
        with pytest.raises(ValueError, match="non-numeric or infinite"):
            model3.fit(X_bad, y_bad, method="normal_equation")
        # Non-numeric y
        X_good = np.array([[1, 2], [3, 4]])
        y_bad2 = np.array([1, np.inf])
        model4 = LinearRegression()
        with pytest.raises(ValueError, match="non-numeric or infinite"):
            model4.fit(X_good, y_bad2, method="normal_equation")
        # Wrong method with valid y
        y_good = np.array([1, 2])
        model5 = LinearRegression()
        with pytest.raises(ValueError, match="method must be 'gradient_descent' or 'normal_equation'"):
            model5.fit(X_good, y_good, method="unsupported")

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
        # Create a simple feature matrix
        X = np.array([[10, 20], [30, 40], [50, 60]])
        # Call the private method directly
        X_with_intercept = model._add_intercept(X)
        # Check shape: should have one more column
        assert X_with_intercept.shape == (3, 3)
        # Check that the first column is all ones
        assert np.all(X_with_intercept[:, 0] == 1)
        # Check that the remaining columns match the original X
        assert np.all(X_with_intercept[:, 1:] == X)

    def test_simple_linear_regression(self, model):
        """Test simple linear regression on known data."""
        # Create noise-free data: y = 2x + 1
        X = np.array([[1], [2], [3], [4], [5]])
        y = 2 * X.flatten() + 1
        model.fit(X, y, method="gradient_descent")
        y_pred = model.predict(X)
        # Predictions should be very close to y
        assert np.allclose(y_pred, y, atol=1e-1)
        # Weights: intercept (weights_[0]), slope (weights_[1])
        assert np.isclose(model.weights_[0], 1, atol=1e-1)
        assert np.isclose(model.weights_[1], 2, atol=1e-1)
        # R² score should be very close to 1
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.999

    def test_multiple_linear_regression(self, model):
        """Test multiple linear regression."""
        # Create noise-free data: y = 3*x1 + 2*x2 + 5
        X = np.array([[1, 2], [2, 1], [3, 4], [4, 3], [5, 5]])
        y = 3 * X[:, 0] + 2 * X[:, 1] + 5
        model.fit(X, y, method="gradient_descent")
        y_pred = model.predict(X)
        # Predictions should be very close to y
        assert np.allclose(y_pred, y, atol=1.0)
        # Weights: intercept (weights_[0]), x1 (weights_[1]), x2 (weights_[2])
        assert np.isclose(model.weights_[0], 5, atol=1.0)
        assert np.isclose(model.weights_[1], 3, atol=1.0)
        assert np.isclose(model.weights_[2], 2, atol=1.0)
        # R² score should be very close to 1
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.99

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

    def test_input_validation_and_warnings(self, model, synthetic_data, one_d_array, nan_2d, inf_2d, mismatched_feature_count_2d):
        """Test input validation and warnings in fit and predict using fixtures."""
        X, y = synthetic_data
        # 1. Large learning_rate warning
        with pytest.warns(UserWarning, match="Large learning_rate may cause convergence issues"):
            LinearRegression(learning_rate=2.0)
        # 2. Few n_iterations warning
        with pytest.warns(UserWarning, match="Very few iterations may not converge"):
            LinearRegression(n_iterations=5)
        # 3. Non-bool verbose
        with pytest.raises(TypeError, match="verbose must be a boolean value"):
            LinearRegression(verbose="yes")
        # 4. X not 2D
        y1 = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="X must be a 2D array"):
            model.fit(one_d_array, y1)
        # 5. y not 1D
        X2 = np.array([[1, 2], [3, 4]])
        y2 = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="y must be a 1D array"):
            model.fit(X2, y2)
        # 6. Mismatched sample counts
        X3 = np.array([[1, 2], [3, 4]])
        y3 = np.array([1])
        with pytest.raises(ValueError, match="Number of samples in X and y must be equal"):
            model.fit(X3, y3)
        # 7. <2 samples
        X4 = np.array([[1, 2]])
        y4 = np.array([1])
        with pytest.raises(ValueError, match="At least 2 data points are required"):
            model.fit(X4, y4)
        # 8. <1 feature
        X5 = np.empty((2, 0))
        y5 = np.array([1, 2])
        with pytest.raises(ValueError, match="At least 1 feature is required"):
            model.fit(X5, y5)
        # 9. Non-numeric/infinite X (NaN)
        y6 = np.array([1, 2])
        with pytest.raises(ValueError, match="X contains non-numeric or infinite values"):
            model.fit(nan_2d, y6)
        # 10. Non-numeric/infinite X (inf)
        with pytest.raises(ValueError, match="X contains non-numeric or infinite values"):
            model.fit(inf_2d, y6)
        # 11. Non-numeric/infinite y
        X7 = np.array([[1, 2], [3, 4]])
        y7 = np.array([1, np.inf])
        with pytest.raises(ValueError, match="y contains non-numeric or infinite values"):
            model.fit(X7, y7)
        # 12. n_samples < n_features + 1 warning
        y8 = np.array([1, 2])
        with pytest.warns(UserWarning, match=r"Number of samples \(2\) should be greater than number of features \(3\)"):
            model.fit(mismatched_feature_count_2d, y8)
        # 13. Unsupported method
        with pytest.raises(ValueError, match="method must be 'gradient_descent' or 'normal_equation'"):
            model.fit(X, y, method="unsupported")
        # 14. TypeError in predict for non-numeric X
        model.fit(X, y)
        X_bad = X.astype(object)
        X_bad[0, 0] = None
        with pytest.raises(ValueError, match="non-numeric or infinite"):
            model.predict(X_bad)
        # 15. TypeError branch in predict for non-numeric X (guaranteed)
        # (Intentionally left blank; cannot reliably cover with numpy)
