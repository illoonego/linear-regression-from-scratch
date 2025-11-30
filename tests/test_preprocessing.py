import numpy as np
import pytest

from linear_regression.preprocessing import StandardScaler


def test_fit_and_transform_normal_data(synthetic_data):
    X, _ = synthetic_data
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    # Check mean and std
    np.testing.assert_allclose(np.mean(X_scaled, axis=0), [0], atol=1e-7)
    np.testing.assert_allclose(np.std(X_scaled, axis=0), [1], atol=1e-7)


def test_constant_feature_handling(constant_feature_2d):
    X = constant_feature_2d
    scaler = StandardScaler()
    scaler.fit(X)
    # std for first column should be 1.0 (replaced)
    assert scaler.std_[0] == 1.0
    X_scaled = scaler.transform(X)
    # All values in first column should be 0 after scaling
    assert np.allclose(X_scaled[:, 0], 0)


def test_transform_before_fit_raises():
    X = np.array([[1, 2], [3, 4]])
    scaler = StandardScaler()
    with pytest.raises(RuntimeError, match="Scaler must be fitted before transform"):
        scaler.transform(X)


def test_transform_mismatched_feature_count_raises(synthetic_data, mismatched_feature_count_2d):
    X, _ = synthetic_data
    scaler = StandardScaler()
    scaler.fit(X)
    X_bad = mismatched_feature_count_2d
    with pytest.raises(ValueError, match="Input data must have 1 features"):
        scaler.transform(X_bad)


def test_validate_input_non_2d_raises(one_d_array):
    scaler = StandardScaler()
    X = one_d_array
    with pytest.raises(ValueError, match="Expected 2D array"):
        scaler._validate_input(X)


def test_validate_input_nan_inf_raises(nan_2d, inf_2d):
    scaler = StandardScaler()
    X_nan = nan_2d
    X_inf = inf_2d
    with pytest.raises(ValueError, match="Input data contains NaN or infinite values"):
        scaler._validate_input(X_nan)
    with pytest.raises(ValueError, match="Input data contains NaN or infinite values"):
        scaler._validate_input(X_inf)
