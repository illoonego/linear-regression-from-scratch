# API Reference

## linear_regression.models.linear_regression

### LinearRegression
```python
class LinearRegression(learning_rate=0.01, n_iterations=1000, fit_intercept=True)
```
Implements linear regression using gradient descent.

**Parameters:**
- `learning_rate` (float, default=0.01): Step size for gradient descent.
- `n_iterations` (int, default=1000): Number of iterations for gradient descent.
- `fit_intercept` (bool, default=True): Whether to include an intercept term.

**Note:** Both gradient descent and normal equation methods are implemented.

**Attributes:**
- `weights_`: Model coefficients (and intercept if enabled)
- `cost_history_`: List of cost values per iteration (gradient descent)
- `is_fitted_`: True if model has been fitted
- `n_features_`: Number of features in training data

**Methods:**

- `fit(X, y, method="gradient_descent"|"normal_equation")`
	- **Parameters:**
		- `X` (np.ndarray, shape (n_samples, n_features)): Training data
		- `y` (np.ndarray, shape (n_samples,)): Target values
		- `method` (str): Optimization method
	- **Returns:** None
	- **Raises:**
		- ValueError: If input shapes are invalid or method is unsupported
		- NotImplementedError: If method is "normal_equation"

- `predict(X)`
	- **Parameters:**
		- `X` (np.ndarray, shape (n_samples, n_features)): Data to predict
	- **Returns:** np.ndarray, shape (n_samples,)
	- **Raises:**
		- ValueError: If model is not fitted or input shape is invalid

- `_add_intercept(X)`
	- **Parameters:**
		- `X` (np.ndarray): Feature matrix
	- **Returns:** np.ndarray

**Limitations:**
- Both gradient descent and normal equation are implemented.
- Does not support categorical features.
- No built-in regularization.
**Example:**
```python
y = np.array([2, 4, 6, 8, 10])
predictions = model.predict(X)

import numpy as np
from linear_regression import LinearRegression
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])
model = LinearRegression()
model.fit(X, y, method="gradient_descent")
predictions = model.predict(X)
```

---

## linear_regression.models.polynomial_regression

### PolynomialRegression
```python
class PolynomialRegression(degree=2, learning_rate=0.01, n_iterations=1000, fit_intercept=True)
```
Implements polynomial regression by expanding features and fitting a LinearRegression model.

**Parameters:**
- `degree` (int, default=2): Degree of polynomial features
- `learning_rate` (float, default=0.01): Step size for gradient descent
- `n_iterations` (int, default=1000): Number of iterations for gradient descent
- `fit_intercept` (bool, default=True): Whether to include an intercept term

**Attributes:**
- `linear_model_`: Underlying LinearRegression instance
- `is_fitted_`: True if model has been fitted
- `n_features_`: Number of original features

**Methods:**
- `fit(X, y, method="gradient_descent"|"normal_equation")`
- `predict(X)`
- `_create_polynomial_features(X)`

**Limitations:**
- No built-in regularization
- High-degree polynomials may overfit (warning issued)

**Example:**
```python
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])
model = PolynomialRegression(degree=2)
model.fit(X, y)
predictions = model.predict(X)
```

---

## linear_regression.preprocessing

### StandardScaler
```python
class StandardScaler()
```
Standardizes features by removing the mean and scaling to unit variance.

**Parameters:** None

**Attributes:**
- `mean_`: Feature means
- `std_`: Feature standard deviations
- `is_fitted_`: True if scaler has been fitted
- `n_features_`: Number of features

**Methods:**

- `fit(X)`
	- **Parameters:**
		- `X` (np.ndarray, shape (n_samples, n_features)): Data to fit scaler
	- **Returns:** None
	- **Raises:**
		- ValueError: If input is not 2D or contains non-numeric data

- `transform(X)`
	- **Parameters:**
		- `X` (np.ndarray, shape (n_samples, n_features)): Data to transform
	- **Returns:** np.ndarray, shape (n_samples, n_features)
	- **Raises:**
		- ValueError: If scaler is not fitted or input shape is invalid

**Limitations:**
- Only supports numeric features.
**Example:**
```python
import numpy as np
from linear_regression import StandardScaler
X = np.array([[1, 2], [3, 4], [5, 6]])
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
```

---

## linear_regression.utils

### train_test_split
```python
train_test_split(X, y, test_size=0.2, random_state=None)
```
Splits arrays or matrices into random train and test subsets.

**Parameters:**
- `X` (np.ndarray, shape (n_samples, n_features)): Features
- `y` (np.ndarray, shape (n_samples,)): Targets
- `test_size` (float, default=0.2): Fraction of data for test set
- `random_state` (int, optional): Seed for reproducibility

**Returns:**
- `X_train` (np.ndarray)
- `X_test` (np.ndarray)
- `y_train` (np.ndarray)
- `y_test` (np.ndarray)

**Raises:**
- ValueError: If input shapes are invalid or test_size is out of bounds

**Returns:** `X_train, X_test, y_train, y_test`

**Example:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
## linear_regression.metrics

### r2_score, mean_squared_error, mean_absolute_error
All metrics are fully implemented and tested. See README for usage examples.

## Error Handling

The classes will raise appropriate exceptions for:
- Mismatched array dimensions
- Non-numeric data
- Singular matrices (normal equation)
- Invalid parameters

## Performance Considerations

- **Normal Equation**: Best for small to medium datasets (< 10,000 features)
- **Gradient Descent**: Better for large datasets
- **Feature Scaling**: Recommended for gradient descent
- **Polynomial Degree**: Higher degrees increase overfitting risk

## Notes
- All dependencies are managed via `pyproject.toml` (PEP 621)