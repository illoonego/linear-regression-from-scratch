import numpy as np


class StandardScaler:
    """Features scaler from scratch.

    This class standardizes features by removing the mean and scaling to unit variance.

    Example:
            >>> from linear_regression.preprocessing import StandardScaler
            >>> import numpy as np
            >>>
            >>> # Create sample data
            >>> X = np.array([[1, 2], [3, 4], [5, 6]])
            >>>
            >>> # Create and fit scaler
            >>> scaler = StandardScaler()
            >>> scaler.fit(X)
            >>>
            >>> # Transform data
            >>> X_scaled = scaler.transform(X)
    """

    def __init__(self):
        """Initialize StandardScaler."""
        self.mean_ = None
        self.std_ = None
        self.is_fitted_ = False
        self.n_features_ = None

    def _validate_input(self, X, check_is_fitted=False):
        """Validate input data.

        Args:
                X (np.ndarray): Input data
                check_is_fitted (bool): Whether to check if the scaler is fitted

        Returns:
                np.ndarray: Validated input data
        """
        # Convert to numpy array if not already
        X = np.asarray(X)

        # Check dimensions
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")

        # Check if fitted
        if check_is_fitted and not self.is_fitted_:
            raise RuntimeError("Scaler must be fitted before transform.")

        # Check feature count (only if fitted)
        if check_is_fitted and X.shape[1] != self.n_features_:
            raise ValueError(f"Input data must have {self.n_features_} features, got {X.shape[1]} instead.")

        # Check for NaN or infinite values
        if not np.isfinite(X).all():
            raise ValueError("Input data contains NaN or infinite values.")

        return X

    def fit(self, X):
        """Compute the mean and std to be used for later scaling.

        Args:
                X (np.ndarray): Input data of shape (n_samples, n_features)
        """
        # Validate input
        X = self._validate_input(X, check_is_fitted=False)

        # Compute stats
        self.mean_ = np.mean(X, axis=0)  # Compute mean for each feature with axis=0
        self.std_ = np.std(X, axis=0)  # Compute std for each feature with axis=0

        # Store number of features
        self.n_features_ = X.shape[1]

        # Handle zero standard deviation (constant features)
        # Replace std=0 with 1.0 to avoid division by zero in transform
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)

        # Mark as fitted
        self.is_fitted_ = True

        return self

    def transform(self, X):
        """Perform standardization by centering and scaling.

        Args:
                X (np.ndarray): Input data of shape (n_samples, n_features)

        Returns:
                np.ndarray: Standardized data of shape (n_samples, n_features)
        """
        # Validate input
        X = self._validate_input(X, check_is_fitted=True)

        # Apply standardization formula
        X_scaled = (X - self.mean_) / self.std_

        return X_scaled
