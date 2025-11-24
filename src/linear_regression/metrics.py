"""Metrics Module for Linear Regression.
This module provides various evaluation metrics for assessing the performance
of linear regression models."""

import numpy as np


def r2_score(y_true, y_pred):
    """Calculate R² score (coefficient of determination).

    Args:
        y_true (np.ndarray): True targets of shape (n_samples,)
        y_pred (np.ndarray): Predicted targets of shape (n_samples,)

    Returns:
        float: R² score
    """

    # ===== INPUT VALIDATION =====
    y_true, y_pred = _validate_input(y_true, y_pred)

    # ===== R² CALCULATION =====
    # Calculate residual sum of squares
    rss = np.sum((y_true - y_pred) ** 2)

    # Calculate total sum of squares
    tss = np.sum((y_true - np.mean(y_true)) ** 2)

    # Return R² score
    if tss == 0:
        return 1.0 if rss == 0 else 0.0  # Handle edge case where y_true is constant

    return 1 - (rss / tss)


def mean_squared_error(y_true, y_pred):
    """Calculate mean squared error.

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values

    Returns:
        float: Mean squared error
    """

    # ===== INPUT VALIDATION =====
    y_true, y_pred = _validate_input(y_true, y_pred)

    # ===== MSE CALCULATION =====
    # Return mean squared error
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    """Calculate mean absolute error.

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values

    Returns:
        float: Mean absolute error
    """

    # ===== INPUT VALIDATION =====
    # Convert to numpy arrays
    y_true, y_pred = _validate_input(y_true, y_pred)

    # ===== MAE CALCULATION =====
    # Return mean absolute error
    return np.mean(np.abs(y_true - y_pred))


def _validate_input(y_true, y_pred):
    """Validate input arrays for metrics functions.

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values

    Raises:
        ValueError: If input arrays have different shapes or are empty
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    # Check edge case of empty arrays
    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty")

    return y_true, y_pred
