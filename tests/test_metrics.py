import numpy as np
import pytest

from linear_regression.metrics import mean_absolute_error, mean_squared_error, r2_score


# === R² SCORE TESTS ===
# Test perfect prediction
def test_r2_score_perfect_fit(perfect_fit_arrays):
    y_true, y_pred = perfect_fit_arrays
    assert r2_score(y_true, y_pred) == 1.0


# Test poor prediction
def test_r2_score_poor_fit(worst_fit_arrays):
    y_true, y_pred = worst_fit_arrays
    assert r2_score(y_true, y_pred) < 1.0


# Test edge case when tss is zero
def test_r2_score_constant_true(constant_arrays):
    y_true, y_pred_perfect, y_pred_varied = constant_arrays
    assert r2_score(y_true, y_pred_perfect) == 1.0
    assert r2_score(y_true, y_pred_varied) == 0.0


# Test input validation for r2_score
def test_r2_score_input_validation(mismatched_arrays, empty_arrays):
    y_true, y_pred = mismatched_arrays
    with pytest.raises(ValueError, match="y_true and y_pred must have the same shape"):
        r2_score(y_true, y_pred)

    y_true, y_pred = empty_arrays
    with pytest.raises(ValueError, match="Input arrays cannot be empty"):
        r2_score(y_true, y_pred)


# Test for non-array inputs
def test_r2_score_non_arrays(non_array_inputs):
    y_true, y_pred = non_array_inputs
    assert r2_score(y_true, y_pred) == r2_score(np.array(y_true), np.array(y_pred))


# === MSE & MAE TESTS ===
def test_mse_perfect_fit(perfect_fit_arrays):
    y_true, y_pred = perfect_fit_arrays
    assert mean_squared_error(y_true, y_pred) == 0.0


def test_mae_perfect_fit(perfect_fit_arrays):
    y_true, y_pred = perfect_fit_arrays
    assert mean_absolute_error(y_true, y_pred) == 0.0


def test_mse_worst_fit(worst_fit_arrays):
    y_true, y_pred = worst_fit_arrays
    expected = np.mean((y_true - y_pred) ** 2)
    assert mean_squared_error(y_true, y_pred) == expected


def test_mae_worst_fit(worst_fit_arrays):
    y_true, y_pred = worst_fit_arrays
    expected = np.mean(np.abs(y_true - y_pred))
    assert mean_absolute_error(y_true, y_pred) == expected


def test_mse_constant_arrays(constant_arrays):
    y_true, y_pred_perfect, y_pred_varied = constant_arrays
    assert mean_squared_error(y_true, y_pred_perfect) == 0.0
    expected = np.mean((y_true - y_pred_varied) ** 2)
    assert mean_squared_error(y_true, y_pred_varied) == expected


def test_mae_constant_arrays(constant_arrays):
    y_true, y_pred_perfect, y_pred_varied = constant_arrays
    assert mean_absolute_error(y_true, y_pred_perfect) == 0.0
    expected = np.mean(np.abs(y_true - y_pred_varied))
    assert mean_absolute_error(y_true, y_pred_varied) == expected


def test_mse_input_validation(mismatched_arrays, empty_arrays):
    import pytest

    y_true, y_pred = mismatched_arrays
    with pytest.raises(ValueError, match="y_true and y_pred must have the same shape"):
        mean_squared_error(y_true, y_pred)

    y_true, y_pred = empty_arrays
    with pytest.raises(ValueError, match="Input arrays cannot be empty"):
        mean_squared_error(y_true, y_pred)


def test_mae_input_validation(mismatched_arrays, empty_arrays):
    import pytest

    y_true, y_pred = mismatched_arrays
    with pytest.raises(ValueError, match="y_true and y_pred must have the same shape"):
        mean_absolute_error(y_true, y_pred)

    y_true, y_pred = empty_arrays
    with pytest.raises(ValueError, match="Input arrays cannot be empty"):
        mean_absolute_error(y_true, y_pred)


def test_mse_non_arrays(non_array_inputs):
    y_true, y_pred = non_array_inputs
    assert mean_squared_error(y_true, y_pred) == mean_squared_error(np.array(y_true), np.array(y_pred))


def test_mae_non_arrays(non_array_inputs):
    y_true, y_pred = non_array_inputs
    assert mean_absolute_error(y_true, y_pred) == mean_absolute_error(np.array(y_true), np.array(y_pred))
