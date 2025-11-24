"""Data splitting utility functions for linear regression package."""

import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None):
    """Split data into training and testing sets.

    Parameters:
        X (array-like): Feature data.
        y (array-like): Target data.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int, optional): Seed for random number generator for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test: Split datasets.
    """

    # Validation of inputs
    # Check if X and y are array-like
    X = np.array(X)
    y = np.array(y)

    # Check if X and y have compatible shapes
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have the same number of samples. Got X: {X.shape[0]} and y: {y.shape[0]}.")

    # Check test_size is between 0 and 1
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1.")

    # Calculate split sizes
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)  # round down to nearest integer

    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)

    # Generate shuffled indices
    indices = np.random.permutation(n_samples)  # [0,1,2,...,n_samples-1] shuffled

    # Split indices
    test_indices = indices[:n_test]  # First n_test indices for testing
    train_indices = indices[n_test:]  # Remaining indices for training

    # Create train and test splits
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test
