"""Tests for PolynomialRegression class."""

import numpy as np
import pytest

from linear_regression.models.polynomial_regression import PolynomialRegression


class TestPolynomialRegression:
    """Test cases for PolynomialRegression class."""

    def test_initialization(self):
        # Valid degree
        model = PolynomialRegression(degree=2)
        assert model.degree == 2
        # Invalid degree type
        with pytest.raises(TypeError):
            PolynomialRegression(degree="2")
        # Invalid degree value
        with pytest.raises(ValueError):
            PolynomialRegression(degree=0)
        # Overfitting warning
        with pytest.warns(UserWarning):
            PolynomialRegression(degree=11)

    def test_polynomial_features(self):
        model = PolynomialRegression(degree=2)
        X = np.array([[1, 2], [3, 4]])
        X_poly = model._create_polynomial_features(X)
        # For degree=2 and 2 features, should have 5 features: x0, x1, x0^2, x0*x1, x1^2
        assert X_poly.shape == (2, 5)
        # Check values for first row
        assert np.allclose(X_poly[0], [1, 2, 1, 2, 4])

    def test_fit_and_predict_quadratic(self, quadratic_data):
        X, y = quadratic_data
        model = PolynomialRegression(degree=2)
        model.fit(X, y, method="normal_equation")
        y_pred = model.predict(X)
        # Should fit perfectly (noise-free)
        assert np.allclose(y_pred, y, atol=1e-6)

    def test_fit_and_predict_cubic(self, cubic_data):
        X, y = cubic_data
        model = PolynomialRegression(degree=3)
        model.fit(X, y, method="normal_equation")
        y_pred = model.predict(X)
        assert np.allclose(y_pred, y, atol=1e-6)

    def test_predict_unfitted(self, quadratic_data):
        X, _ = quadratic_data
        model = PolynomialRegression(degree=2)
        with pytest.raises(ValueError, match="fitted"):
            model.predict(X)

    def test_predict_wrong_shape(self, quadratic_data):
        X, y = quadratic_data
        model = PolynomialRegression(degree=2)
        model.fit(X, y)
        # 1D input
        with pytest.raises(ValueError, match="2D array"):
            model.predict(X[:, 0])
        # Wrong feature count
        X_wrong = np.hstack([X, X])
        with pytest.raises(ValueError, match="Input features have"):
            model.predict(X_wrong)

    def test_fit_wrong_shape(self, quadratic_data):
        X, y = quadratic_data
        model = PolynomialRegression(degree=2)
        # 1D X
        with pytest.raises(ValueError, match="2D array"):
            model.fit(X[:, 0], y)
        # 2D y
        y2d = y.reshape(-1, 1)
        with pytest.raises(ValueError, match="1D array"):
            model.fit(X, y2d)
        # Mismatched samples
        with pytest.raises(ValueError, match="equal"):
            model.fit(X, y[:-1])

    def test_overfitting_high_degree(self, quadratic_data):
        X, y = quadratic_data
        with pytest.warns(UserWarning, match="overfitting"):
            PolynomialRegression(degree=15).fit(X, y)
