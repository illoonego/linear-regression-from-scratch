# API Reference

## linear_regression.models

### LinearRegression

```python
class LinearRegression(learning_rate=0.01, n_iterations=1000, fit_intercept=True)
```

Linear regression implementation using gradient descent and normal equation methods.

#### Parameters

- **learning_rate** (*float*, default=0.01): Learning rate for gradient descent optimization
- **n_iterations** (*int*, default=1000): Number of iterations for gradient descent
- **fit_intercept** (*bool*, default=True): Whether to calculate the intercept for this model

#### Attributes

- **weights** (*np.ndarray*): Model parameters (coefficients and intercept)
- **cost_history** (*list*): History of cost function values during training

#### Methods

##### fit(X, y, method='gradient_descent')

Fit the linear regression model to training data.

**Parameters:**
- **X** (*np.ndarray*): Training features of shape (n_samples, n_features)
- **y** (*np.ndarray*): Training targets of shape (n_samples,)
- **method** (*str*, default='gradient_descent'): Optimization method ('gradient_descent' or 'normal_equation')

**Returns:**
- **self**: Returns self for method chaining

##### predict(X)

Make predictions using the trained model.

**Parameters:**
- **X** (*np.ndarray*): Features to predict on of shape (n_samples, n_features)

**Returns:**
- **np.ndarray**: Predictions of shape (n_samples,)

##### score(X, y)

Calculate R² score (coefficient of determination).

**Parameters:**
- **X** (*np.ndarray*): Features of shape (n_samples, n_features)
- **y** (*np.ndarray*): True targets of shape (n_samples,)

**Returns:**
- **float**: R² score

##### mean_squared_error(y_true, y_pred)

Calculate mean squared error.

**Parameters:**
- **y_true** (*np.ndarray*): True values
- **y_pred** (*np.ndarray*): Predicted values

**Returns:**
- **float**: Mean squared error

##### mean_absolute_error(y_true, y_pred)

Calculate mean absolute error.

**Parameters:**
- **y_true** (*np.ndarray*): True values
- **y_pred** (*np.ndarray*): Predicted values

**Returns:**
- **float**: Mean absolute error

#### Example

```python
from linear_regression.models.linear_regression import LinearRegression
import numpy as np

# Create sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create and train model
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y, method='gradient_descent')

# Make predictions
predictions = model.predict(X)

# Evaluate model
r2_score = model.score(X, y)
mse = model.mean_squared_error(y, predictions)
```

### PolynomialRegression

```python
class PolynomialRegression(degree=2, learning_rate=0.01, n_iterations=1000, fit_intercept=True)
```

Polynomial regression implementation that transforms features into polynomial features.

#### Parameters

- **degree** (*int*, default=2): Degree of polynomial features
- **learning_rate** (*float*, default=0.01): Learning rate for gradient descent
- **n_iterations** (*int*, default=1000): Number of iterations for gradient descent
- **fit_intercept** (*bool*, default=True): Whether to calculate the intercept

#### Attributes

- **degree** (*int*): Degree of polynomial features
- **linear_model** (*LinearRegression*): Underlying linear regression model

#### Methods

##### fit(X, y, method='gradient_descent')

Fit the polynomial regression model to training data.

**Parameters:**
- **X** (*np.ndarray*): Training features of shape (n_samples, n_features)
- **y** (*np.ndarray*): Training targets of shape (n_samples,)
- **method** (*str*, default='gradient_descent'): Optimization method

**Returns:**
- **self**: Returns self for method chaining

##### predict(X)

Make predictions using the trained model.

**Parameters:**
- **X** (*np.ndarray*): Features to predict on of shape (n_samples, n_features)

**Returns:**
- **np.ndarray**: Predictions of shape (n_samples,)

##### score(X, y)

Calculate R² score (coefficient of determination).

**Parameters:**
- **X** (*np.ndarray*): Features of shape (n_samples, n_features)
- **y** (*np.ndarray*): True targets of shape (n_samples,)

**Returns:**
- **float**: R² score

#### Example

```python
from linear_regression.models.polynomial_regression import PolynomialRegression
import numpy as np

# Create sample data (quadratic relationship)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])  # y = x²

# Create and train model
model = PolynomialRegression(degree=2)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Evaluate model
r2_score = model.score(X, y)
```

## Usage Patterns

### Simple Linear Regression

```python
# For simple linear regression (one feature)
from linear_regression.models import LinearRegression

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X_test)
```

### Multiple Linear Regression

```python
# For multiple features
X_multi = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([5, 8, 11, 14])

model = LinearRegression()
model.fit(X_multi, y)
```

### Polynomial Features

```python
# For non-linear relationships
from linear_regression.models import PolynomialRegression

# Quadratic fit
model = PolynomialRegression(degree=2)
model.fit(X, y)

# Cubic fit
model_cubic = PolynomialRegression(degree=3)
model_cubic.fit(X, y)
```

### Method Comparison

```python
# Compare gradient descent vs normal equation
model_gd = LinearRegression()
model_gd.fit(X, y, method='gradient_descent')

model_ne = LinearRegression()
model_ne.fit(X, y, method='normal_equation')
```

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