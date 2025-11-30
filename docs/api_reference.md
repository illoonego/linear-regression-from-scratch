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

# Install from PyPI
pip install linear-regression-from-scratch

from linear_regression.models.linear_regression import LinearRegression
import numpy as np
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])
model = LinearRegression()
model.fit(X, y, method="gradient_descent")
predictions = model.predict(X)
```

---

## linear_regression.models.polynomial_regression

### PolynomialRegression
**Not implemented.** All methods are stubs and raise NotImplementedError.

**Limitations:**
- All methods raise NotImplementedError.

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
from linear_regression.preprocessing import StandardScaler
import numpy as np
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

### r2_score
```python
r2_score(y_true, y_pred)
```
Returns the coefficient of determination $R^2$.

**Parameters:**
- `y_true` (np.ndarray, shape (n_samples,)): True values
- `y_pred` (np.ndarray, shape (n_samples,)): Predicted values

**Returns:** float

**Raises:**
- ValueError: If input shapes are invalid

### mean_squared_error
```python
mean_squared_error(y_true, y_pred)
```

**Parameters:**
- `y_true` (np.ndarray, shape (n_samples,)): True values
- `y_pred` (np.ndarray, shape (n_samples,)): Predicted values

**Returns:** float

**Raises:**
- ValueError: If input shapes are invalid

### mean_absolute_error
```python
mean_absolute_error(y_true, y_pred)
```
Returns the mean absolute error.

**Parameters:**
- `y_true` (np.ndarray, shape (n_samples,)): True values
- `y_pred` (np.ndarray, shape (n_samples,)): Predicted values

**Returns:** float

**Raises:**
- ValueError: If input shapes are invalid

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