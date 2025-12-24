# Mathematical Background

## Linear Regression
### System of Linear Equations

Linear regression is fundamentally a system of equations—one for each data point. For $m$ data points and $n$ features:

```math
\begin{align*}
y^{(1)} &= \beta_0 + \beta_1 x_1^{(1)} + \beta_2 x_2^{(1)} + \ldots + \beta_n x_n^{(1)} + \epsilon^{(1)} \\
y^{(2)} &= \beta_0 + \beta_1 x_1^{(2)} + \beta_2 x_2^{(2)} + \ldots + \beta_n x_n^{(2)} + \epsilon^{(2)} \\
&\vdots \\
y^{(m)} &= \beta_0 + \beta_1 x_1^{(m)} + \beta_2 x_2^{(m)} + \ldots + \beta_n x_n^{(m)} + \epsilon^{(m)}
\end{align*}
```

### Matric Form
To handle many samples efficiently, we rewrite linear regression using matrices:

#### Feature Matrix
For $m$ samples and $n$ features, the feature matrix $\mathbf{X}$ is:
```math
\mathbf{X} =
\begin{bmatrix}
x_{11} & x_{12} & \dots & x_{1n} \\
x_{21} & x_{22} & \dots & x_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
x_{m1} & x_{m2} & \dots & x_{mn}
\end{bmatrix}
```
Each row represents a sample, each column a feature.

#### Adding Intercept Column
To include the intercept term $\beta_0$, we prepend a column of ones to $\mathbf{X}$, forming $\mathbf{X'}$:
```math
\mathbf{X}' =
\begin{bmatrix}
1 & x_{11} & x_{12} & \dots & x_{1n} \\
1 & x_{21} & x_{22} & \dots & x_{2n} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{m1} & x_{m2} & \dots & x_{mn}
\end{bmatrix}
```
This allows the model to learn an intercept.

#### Coefficient Vector
The coefficients (including intercept) are stored in a vector $\boldsymbol{\beta}$:
```math
\boldsymbol{\beta} = \begin{bmatrix}
  \beta_0 \\
  \beta_1 \\
  \beta_2 \\
  \vdots \\
  \beta_n
\end{bmatrix}
```

#### Target Vector
The target values are stored in a vector $\mathbf{y}$:
```math
\mathbf{y} = \begin{bmatrix}
  y_1 \\
  y_2 \\
  \vdots \\
  y_m
\end{bmatrix}
```

#### Matrix Form of the Model
The matrix form of linear regression is:
```math
\mathbf{y} = \mathbf{X'} \boldsymbol{\beta} + \boldsymbol{\epsilon}
```
Where:
- $\mathbf{y}$ is the vector of all target values
- $\mathbf{X'}$ is the feature matrix with intercept column
- $\boldsymbol{\beta}$ is the vector of all coefficients
- $\boldsymbol{\epsilon}$ is the vector of errors

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

Polynomial regression extends linear regression by expanding the feature space to include powers and cross-terms of the original features. For degree $d$ and $n$ features, the number of polynomial features grows combinatorially.

### Polynomial Feature Expansion
Given $X$ with $n$ features, polynomial features up to degree $d$ are generated using all combinations (with replacement) of the original features:

```math
\text{For degree } d=2, n=2:
\begin{bmatrix}
x_1 & x_2 \\
\end{bmatrix}
\rightarrow
\begin{bmatrix}
1 & x_1 & x_2 & x_1^2 & x_1 x_2 & x_2^2
\end{bmatrix}
```

### Matrix Form
The expanded feature matrix $\mathbf{X}_{poly}$ is used in the same way as in linear regression:
```math
\mathbf{y} = \mathbf{X}_{poly} \boldsymbol{\beta} + \boldsymbol{\epsilon}
```

### Overfitting Warning
High-degree polynomials can fit training data perfectly but generalize poorly. The implementation issues a warning for degrees > 10.