"""Tests for LinearRegression class."""

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

    def test_fit_gradient_descent(self, model, synthetic_data):
        """Test fitting with gradient descent."""
        # TODO: Test gradient descent fitting
        pass
    
    def test_fit_normal_equation(self, model, synthetic_data):
        """Test fitting with normal equation."""
        # TODO: Test normal equation fitting
        pass
    
    def test_predict(self, model, synthetic_data):
        """Test prediction functionality."""
        # TODO: Test predictions
        pass
    
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