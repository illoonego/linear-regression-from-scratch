"""File to set up test configurations."""

import pytest
import numpy as np
from linear_regression.models.linear_regression import LinearRegression

# Sample fixture for generating synthetic data
@pytest.fixture
def synthetic_data():
    """Generate synthetic linear data for testing."""
    np.random.seed(42) # Reproduce noise
    X = np.array([[1], [2], [3], [4], [5]])
    slope = 2.0
    intercept = 3.0
    y = slope * X.flatten() + intercept + np.random.randn(5) # y = 2x + 3 + noise
    return X, y

# Fixture for a LinearRegression model instance
@pytest.fixture
def model():
    """Create a LinearRegression model instance for testing."""
    return LinearRegression(learning_rate=0.01, n_iterations=1000, fit_intercept=True)

# mismatch in array sizes
@pytest.fixture
def mismatched_arrays():
    """Create mismatched arrays for testing."""
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2])
    return y_true, y_pred

# Empty arrays
@pytest.fixture
def empty_arrays():
    """Create empty arrays for testing."""
    y_true = np.array([])
    y_pred = np.array([])
    return y_true, y_pred

# Perfect prediction arrays
@pytest.fixture
def perfect_fit_arrays():
    """Create arrays for perfect prediction testing."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    return y_true, y_pred

# Worst prediction arrays
@pytest.fixture
def worst_fit_arrays():
    """Create arrays for worst prediction testing."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([5, 4, 3, 2, 1])
    return y_true, y_pred

# Constant arrays
@pytest.fixture
def constant_arrays():
    """Create constant arrays for testing."""
    y_true = np.array([5, 5, 5, 5])
    y_pred_perfect = np.array([5, 5, 5, 5])
    y_pred_varied = np.array([4, 6, 5, 5])
    return y_true, y_pred_perfect, y_pred_varied

# Non-array inputs
@pytest.fixture
def non_array_inputs():
    """Create non-array inputs for testing."""
    y_true = [1, 2, 3]
    y_pred = [4, 5, 6]
    return y_true, y_pred