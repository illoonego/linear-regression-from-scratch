# Mathematical Background

## Linear Regression Theory

### Simple Linear Regression

Simple linear regression models the relationship between a single independent variable $x$ and a dependent variable $y$ using a linear function:

$$y = \beta_0 + \beta_1 x + \epsilon$$

Where:
- $\beta_0$ is the y-intercept (bias term)
- $\beta_1$ is the slope (weight)
- $\epsilon$ is the error term

### Multiple Linear Regression

Multiple linear regression extends this to multiple independent variables:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n + \epsilon$$

In matrix form:
$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

Where:
- $\mathbf{y}$ is the target vector of shape $(m, 1)$
- $\mathbf{X}$ is the feature matrix of shape $(m, n+1)$ (including intercept column)
- $\boldsymbol{\beta}$ is the parameter vector of shape $(n+1, 1)$
- $\boldsymbol{\epsilon}$ is the error vector

### Cost Function

The most commonly used cost function is Mean Squared Error (MSE):

$$J(\boldsymbol{\beta}) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\boldsymbol{\beta}}(\mathbf{x}^{(i)}) - y^{(i)})^2$$

In matrix form:
$$J(\boldsymbol{\beta}) = \frac{1}{2m} (\mathbf{X}\boldsymbol{\beta} - \mathbf{y})^T (\mathbf{X}\boldsymbol{\beta} - \mathbf{y})$$

## Optimization Methods

### 1. Normal Equation (Closed-Form Solution)

The optimal parameters can be found analytically by setting the gradient to zero:

$$\boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

**Advantages:**
- Exact solution
- No need to choose learning rate
- No iterations required

**Disadvantages:**
- Computationally expensive for large datasets ($O(n^3)$)
- Requires matrix inversion
- May be numerically unstable if $\mathbf{X}^T \mathbf{X}$ is singular

### 2. Gradient Descent

Iteratively updates parameters in the direction of steepest descent:

$$\boldsymbol{\beta} := \boldsymbol{\beta} - \alpha \nabla J(\boldsymbol{\beta})$$

The gradient is:
$$\nabla J(\boldsymbol{\beta}) = \frac{1}{m} \mathbf{X}^T (\mathbf{X}\boldsymbol{\beta} - \mathbf{y})$$

**Algorithm:**
1. Initialize $\boldsymbol{\beta}$ randomly
2. Repeat until convergence:
   - Calculate predictions: $\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\beta}$
   - Calculate error: $\mathbf{e} = \hat{\mathbf{y}} - \mathbf{y}$
   - Calculate gradient: $\nabla J = \frac{1}{m} \mathbf{X}^T \mathbf{e}$
   - Update parameters: $\boldsymbol{\beta} := \boldsymbol{\beta} - \alpha \nabla J$

**Advantages:**
- Works well with large datasets
- Simple to implement
- Memory efficient

**Disadvantages:**
- Requires choosing learning rate
- May converge slowly
- Can get stuck in local minima (though not an issue for linear regression)

### 3. Stochastic Gradient Descent (SGD)

Updates parameters using one training example at a time:

$$\boldsymbol{\beta} := \boldsymbol{\beta} - \alpha \nabla J^{(i)}(\boldsymbol{\beta})$$

Where $\nabla J^{(i)}(\boldsymbol{\beta}) = (\hat{y}^{(i)} - y^{(i)}) \mathbf{x}^{(i)}$

## Polynomial Regression

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