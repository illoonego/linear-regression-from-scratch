# Mathematical Background
# Linear Regression as a System of Equations

In practice, linear regression is not just a single equation, but a system of equations—one for each data point. For $m$ data points, we have:

# Mathematical Background

## Linear Regression: System of Equations and Matrix Form

Linear regression is fundamentally a system of equations—one for each data point. For $m$ data points and $n$ features:

$$
\begin{align*}
y^{(1)} &= \beta_0 + \beta_1 x_1^{(1)} + \beta_2 x_2^{(1)} + \ldots + \beta_n x_n^{(1)} + \epsilon^{(1)} \\
y^{(2)} &= \beta_0 + \beta_1 x_1^{(2)} + \beta_2 x_2^{(2)} + \ldots + \beta_n x_n^{(2)} + \epsilon^{(2)} \\
&\vdots \\
y^{(m)} &= \beta_0 + \beta_1 x_1^{(m)} + \beta_2 x_2^{(m)} + \ldots + \beta_n x_n^{(m)} + \epsilon^{(m)}
\end{align*}
$$

This system is compactly represented in matrix form:

$$
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}
$$

Where:
- $\mathbf{y}$: Target vector $(m, 1)$
- $\mathbf{X}$: Feature matrix $(m, n+1)$ (first column is ones for intercept)
- $\boldsymbol{\beta}$: Parameter vector $(n+1, 1)$
- $\boldsymbol{\epsilon}$: Error vector $(m, 1)$

**Visualization:**
- Scatter plot with each point representing an equation.
- Animation: Show how all points contribute to the system.

## Why Matrix Form?

Matrix notation enables:
- Vectorized, efficient computation
- Clear expression of optimization methods
- Scalability to large datasets

## Cost Function

The most common cost function is Mean Squared Error (MSE):

$$J(\boldsymbol{\beta}) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\boldsymbol{\beta}}(\mathbf{x}^{(i)}) - y^{(i)})^2$$

In matrix form:
$$J(\boldsymbol{\beta}) = \frac{1}{2m} (\mathbf{X}\boldsymbol{\beta} - \mathbf{y})^T (\mathbf{X}\boldsymbol{\beta} - \mathbf{y})$$

## Optimization Methods

### Normal Equation (Closed-Form Solution)

Analytically solve for parameters:
$$
\boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

**Pros:** Exact, no learning rate, no iterations
**Cons:** Slow for large $n$, requires matrix inversion, unstable if $\mathbf{X}^T\mathbf{X}$ is singular

### Gradient Descent

Iteratively update parameters:
$$
\boldsymbol{\beta} := \boldsymbol{\beta} - \alpha \nabla J(\boldsymbol{\beta})
$$
where
$$
\nabla J(\boldsymbol{\beta}) = \frac{1}{m} \mathbf{X}^T (\mathbf{X}\boldsymbol{\beta} - \mathbf{y})
$$

**Algorithm:**
1. Initialize $\boldsymbol{\beta}$
2. Repeat:
   - Predict: $\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\beta}$
   - Error: $\mathbf{e} = \hat{\mathbf{y}} - \mathbf{y}$
   - Gradient: $\nabla J = \frac{1}{m} \mathbf{X}^T \mathbf{e}$
   - Update: $\boldsymbol{\beta} := \boldsymbol{\beta} - \alpha \nabla J$

**Visualization:**
- Contour plot of $J(\boldsymbol{\beta})$ with arrows for gradient steps
- Animation: Show parameter updates moving toward minimum

### Stochastic Gradient Descent (SGD)

Update using one sample at a time:
$$
\boldsymbol{\beta} := \boldsymbol{\beta} - \alpha \nabla J^{(i)}(\boldsymbol{\beta})
$$
where $\nabla J^{(i)}(\boldsymbol{\beta}) = (\hat{y}^{(i)} - y^{(i)}) \mathbf{x}^{(i)}$

## Polynomial Regression

Transform features to fit non-linear relationships:
$$
y = \beta_0 + \beta_1 x + \beta_2 x^2 + \ldots + \beta_d x^d + \epsilon
$$

For multiple features, create polynomial and interaction terms.

## Model Evaluation

### $R^2$ (Coefficient of Determination)
$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

### Mean Squared Error (MSE)
$$
MSE = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
$$

### Mean Absolute Error (MAE)
$$
MAE = \frac{1}{m} \sum_{i=1}^{m} |y_i - \hat{y}_i|
$$

## Implementation Considerations

### Feature Scaling

For gradient descent, scale features for faster convergence:
- **Standardization**: $x' = \frac{x - \mu}{\sigma}$
- **Normalization**: $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$

### Regularization

Reduce overfitting, especially for polynomials:

**Ridge (L2):**
$$J(\boldsymbol{\beta}) = \frac{1}{2m} \|\mathbf{X}\boldsymbol{\beta} - \mathbf{y}\|^2 + \lambda \|\boldsymbol{\beta}\|^2$$

**Lasso (L1):**
$$J(\boldsymbol{\beta}) = \frac{1}{2m} \|\mathbf{X}\boldsymbol{\beta} - \mathbf{y}\|^2 + \lambda \|\boldsymbol{\beta}\|_1$$

### Bias-Variance Tradeoff

- **High Bias, Low Variance**: Underfitting
- **Low Bias, High Variance**: Overfitting
- **Goal**: Balance for generalization
Polynomial regression extends linear regression by transforming features:

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \ldots + \beta_d x^d + \epsilon$$

For multiple features, we can create polynomial features:
- $x_1, x_2$ (original features)
- $x_1^2, x_2^2$ (squared terms)
- $x_1 x_2$ (interaction terms)
- Higher-order terms...

## Model Evaluation

### R-squared (Coefficient of Determination)

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

Where:
- $SS_{res}$ is the sum of squared residuals
- $SS_{tot}$ is the total sum of squares
- $\bar{y}$ is the mean of observed values

### Mean Squared Error (MSE)

$$MSE = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

### Mean Absolute Error (MAE)

$$MAE = \frac{1}{m} \sum_{i=1}^{m} |y_i - \hat{y}_i|$$

## Implementation Considerations

### Feature Scaling

For gradient descent, it's often beneficial to scale features:
- **Standardization**: $x' = \frac{x - \mu}{\sigma}$
- **Normalization**: $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$

### Regularization

To prevent overfitting, especially in polynomial regression:

**Ridge Regression (L2):**
$$J(\boldsymbol{\beta}) = \frac{1}{2m} \|\mathbf{X}\boldsymbol{\beta} - \mathbf{y}\|^2 + \lambda \|\boldsymbol{\beta}\|^2$$

**Lasso Regression (L1):**
$$J(\boldsymbol{\beta}) = \frac{1}{2m} \|\mathbf{X}\boldsymbol{\beta} - \mathbf{y}\|^2 + \lambda \|\boldsymbol{\beta}\|_1$$

### Bias-Variance Tradeoff

- **High Bias, Low Variance**: Underfitting (model too simple)
- **Low Bias, High Variance**: Overfitting (model too complex)
- **Goal**: Find the right balance for good generalization