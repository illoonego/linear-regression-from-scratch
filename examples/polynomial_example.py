"""Polynomial Regression Example.

This example demonstrates how to use the PolynomialRegression class
for fitting nonlinear data and visualizing the model fit.
"""

import numpy as np
import matplotlib.pyplot as plt

from linear_regression.models.polynomial_regression import PolynomialRegression
from linear_regression.preprocessing import StandardScaler
from linear_regression.metrics import r2_score, mean_squared_error
from linear_regression.utils import train_test_split


def example_polynomial_fit():
    """Example of polynomial regression with synthetic nonlinear data."""
    print("\nPolynomial Regression Example")
    print("-" * 40)

    # Generate synthetic nonlinear data
    np.random.seed(42)
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_true = 0.5 * X.flatten() ** 3 - X.flatten() ** 2 + 2 * X.flatten() + 3 + np.random.randn(100) * 3
    print(f"Data points: {len(X)}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)

    # Preprocess features
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train polynomial regression model
    degree = 3
    model = PolynomialRegression(degree=degree, learning_rate=0.01, n_iterations=2000, fit_intercept=True)
    print(f"\nTraining PolynomialRegression (degree={degree})...")
    model.fit(X_train_scaled, y_train, method="gradient_descent")
    print("Training completed!")

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate performance
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nResults:")
    print(f"Testing Set - R² Score: {r2:.4f}, MSE: {mse:.4f}")

    # Visualize model fit
    X_plot = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    X_plot_scaled = scaler.transform(X_plot)
    y_plot = model.predict(X_plot_scaled)

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y_true, color="blue", label="Data Points", alpha=0.6)
    plt.plot(X_plot, y_plot, color="red", label=f"Polynomial Fit (degree={degree})", linewidth=2)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Polynomial Regression Fit")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    """Run polynomial regression example."""
    print("\nPolynomial Regression from Scratch - Example")
    print("=" * 50)
    example_polynomial_fit()


if __name__ == "__main__":
    main()
