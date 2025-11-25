import numpy as np
import pytest

from linear_regression.utils import train_test_split


def test_train_test_split_basic(synthetic_data):
    X, y = synthetic_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    assert X_train.shape[0] == 4
    assert X_test.shape[0] == 1
    assert y_train.shape[0] == 4
    assert y_test.shape[0] == 1
    # Check reproducibility
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=42)
    np.testing.assert_array_equal(X_train, X_train2)
    np.testing.assert_array_equal(X_test, X_test2)
    np.testing.assert_array_equal(y_train, y_train2)
    np.testing.assert_array_equal(y_test, y_test2)


def test_train_test_split_invalid_shapes(mismatched_arrays):
    y_true, y_pred = mismatched_arrays
    # Use y_true as X, y_pred as y to simulate mismatch
    with pytest.raises(ValueError, match="X and y must have the same number of samples"):
        train_test_split(y_true.reshape(-1, 1), y_pred, test_size=0.2)


def test_train_test_split_invalid_test_size(synthetic_data):
    X, y = synthetic_data
    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        train_test_split(X, y, test_size=0)
    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        train_test_split(X, y, test_size=1)
    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        train_test_split(X, y, test_size=-0.1)


def test_train_test_split_output_types(synthetic_data):
    X, y = synthetic_data
    X_train, X_test, y_train, y_test = train_test_split(X.tolist(), y.tolist(), test_size=0.3, random_state=0)
    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)


def test_train_test_split_edge_case_small_test():
    X = np.arange(5).reshape(-1, 1)
    y = np.arange(5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # With 5 samples and test_size=0.2, n_test should be int(5*0.2)=1
    assert X_test.shape[0] == 1
    assert y_test.shape[0] == 1
    assert X_train.shape[0] == 4
    assert y_train.shape[0] == 4
