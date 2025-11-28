"""Basic Linear Regression Example.

This example demonstrates how to use the LinearRegression class
for simple and multiple linear regression tasks.
"""

import numpy as np

from linear_regression.metrics import mean_squared_error, r2_score
from linear_regression.models.linear_regression import LinearRegression
from linear_regression.preprocessing import StandartScaler
from linear_regression.utils import train_test_split


def example_1d_simple():
    """Example of simple linear regression with 1D data."""
    print("\n1D Simple Linear Regression Example")
    print("-" * 40)

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    X = np.arange(100).reshape(-1, 1)  # reshape for 1D feature
    slope = 2.0
    intercept = 3.0
    y_true = slope * X.flatten() + intercept + np.random.randn(100)  # y = 2x + 3 + noise
    print(f"Data points: {len(X)}")
    print(f"True weights: slope={slope}, intercept={intercept}")

    # Split data into training and testing sets
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)

    # Preprocess features
    print("\nPreprocessing features with StandardScaler...")
    scaler = StandartScaler()
    print("Fitting scaler on training data...")
    scaler.fit(X_train)
    print("Scaler fitted on training data!")

    print("Transforming training and testing data...")
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Data transformation completed!")

    # Train with Gradient Descent
    model_gd = LinearRegression(learning_rate=0.01, n_iterations=1000, fit_intercept=True)
    method_gd = "gradient_descent"
    print(f"\nTraining model with {method_gd.replace('_', ' ').title()}...")
    model_gd.fit(X_train_scaled, y_train, method=method_gd)
    print("Training completed!")

    # Train with Normal Equation
    model_ne = LinearRegression(fit_intercept=True)
    method_ne = "normal_equation"
    print(f"\nTraining model with {method_ne.replace('_', ' ').title()} (closed-form)...")
    model_ne.fit(X_train_scaled, y_train, method=method_ne)
    print("Training completed!")

    # Make predictions
    y_pred_gd = model_gd.predict(X_test_scaled)
    y_pred_ne = model_ne.predict(X_test_scaled)

    # Evaluate performance
    r2_gd = r2_score(y_test, y_pred_gd)
    mse_gd = mean_squared_error(y_test, y_pred_gd)
    r2_ne = r2_score(y_test, y_pred_ne)
    mse_ne = mean_squared_error(y_test, y_pred_ne)

    # Unsclae weights to original scale
    weight_gd_unscaled = model_gd.weights_[1] / scaler.std_[0]
    intercept_gd_unscaled = model_gd.weights_[0] - (weight_gd_unscaled * scaler.mean_[0])
    weight_ne_unscaled = model_ne.weights_[1] / scaler.std_[0]
    intercept_ne_unscaled = model_ne.weights_[0] - (weight_ne_unscaled * scaler.mean_[0])

    print("\nResults (Gradient Descent):")
    print(f"Testing Set - R² Score: {r2_gd:.4f}, MSE: {mse_gd:.4f}")
    print(f"Learned weights (original scale): slope={weight_gd_unscaled:.2f}, intercept={intercept_gd_unscaled:.2f}")

    print("\nResults (Normal Equation):")
    print(f"Testing Set - R² Score: {r2_ne:.4f}, MSE: {mse_ne:.4f}")
    print(f"Learned weights (original scale): slope={weight_ne_unscaled:.2f}, intercept={intercept_ne_unscaled:.2f}")

    print("\nTrue weights:")
    print(f"slope={slope}, intercept={intercept}")


def example_2d_multiple():
    """Example of multiple linear regression with 2D data."""
    print("\n2D Multiple Linear Regression Example")
    print("-" * 40)

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    np.random.seed(42)
    size_sqft = np.random.uniform(800, 2500, 100)  # Real house sizes
    bedrooms = np.random.randint(1, 5, 100)  # Real bedroom counts
    X = np.column_stack((size_sqft, bedrooms))
    noise = np.random.randn(100) * 10000
    price = 150 * size_sqft + 10000 * bedrooms + 20000 + noise  # price = 150*size + 10000*bedrooms + 20000 + noise
    print(f"Data points: {len(X)}")
    print("True weights: size coefficient=150, bedroom coefficient=10000, intercept=20000")

    # Split data into training and testing sets
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, price, test_size=0.2, random_state=42)

    # Preprocess features
    print("\nPreprocessing features with StandardScaler...")
    scaler = StandartScaler()

    print("Fitting scaler on training data...")
    scaler.fit(X_train)

    print("Transforming training and testing data...")
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Data transformation completed!")

    # Train with Gradient Descent
    model_gd = LinearRegression(learning_rate=0.01, n_iterations=1000, fit_intercept=True)
    method_gd = "gradient_descent"
    print(f"\nTraining model with {method_gd.replace('_', ' ').title()}...")
    model_gd.fit(X_train_scaled, y_train, method=method_gd)
    print("Training completed!")

    # Train with Normal Equation
    model_ne = LinearRegression(fit_intercept=True)
    method_ne = "normal_equation"
    print(f"\nTraining model with {method_ne.replace('_', ' ').title()} (closed-form)...")
    model_ne.fit(X_train_scaled, y_train, method=method_ne)
    print("Training completed!")

    # Make predictions
    y_pred_gd = model_gd.predict(X_test_scaled)
    y_pred_ne = model_ne.predict(X_test_scaled)

    # Evaluate performance
    r2_gd = r2_score(y_test, y_pred_gd)
    mse_gd = mean_squared_error(y_test, y_pred_gd)
    r2_ne = r2_score(y_test, y_pred_ne)
    mse_ne = mean_squared_error(y_test, y_pred_ne)

    # Unscale weights to original scale
    weight_gd_unscaled = model_gd.weights_[1] / scaler.std_[0]
    bedroom_gd_unscaled = model_gd.weights_[2] / scaler.std_[1]
    intercept_gd_unscaled = model_gd.weights_[0] - (weight_gd_unscaled * scaler.mean_[0]) - (bedroom_gd_unscaled * scaler.mean_[1])

    weight_ne_unscaled = model_ne.weights_[1] / scaler.std_[0]
    bedroom_ne_unscaled = model_ne.weights_[2] / scaler.std_[1]
    intercept_ne_unscaled = model_ne.weights_[0] - (weight_ne_unscaled * scaler.mean_[0]) - (bedroom_ne_unscaled * scaler.mean_[1])

    print("\nResults (Gradient Descent):")
    print(f"Testing Set - R² Score: {r2_gd:.4f}, MSE: {mse_gd:.4f}")
    print(
        f"Learned weights (original scale): size={weight_gd_unscaled:.2f}, "
        f"bedroom={bedroom_gd_unscaled:.2f}, "
        f"intercept={intercept_gd_unscaled:.2f}"
    )

    print("\nResults (Normal Equation):")
    print(f"Testing Set - R² Score: {r2_ne:.4f}, MSE: {mse_ne:.4f}")
    print(
        f"Learned weights (original scale): size={weight_ne_unscaled:.2f}, "
        f"bedroom={bedroom_ne_unscaled:.2f}, "
        f"intercept={intercept_ne_unscaled:.2f}"
    )

    print("\nTrue weights:")
    print("size=150.00, bedroom=10000.00, intercept=20000.00")


def main():
    """Run basic linear regression examples."""
    import sys

    print("\nLinear Regression from Scratch - Basic Example")
    print("=" * 50)

    # Check command line arguments
    if len(sys.argv) > 1:
        example_type = sys.argv[1].lower()

        if example_type == "1d":
            print("\n🎯 Running 1D example only...")
            example_1d_simple()
        elif example_type == "2d":
            print("\n🎯 Running 2D example only...")
            example_2d_multiple()
        elif example_type == "all":
            print("\n🎯 Running all examples...")
            example_1d_simple()
            example_2d_multiple()
        else:
            print("\n❌ Invalid argument. Use: 1d, 2d, or all")
            print("Usage: python basic_linear_regression.py [1d|2d|all]")
            return
    else:
        # Default behavior: run all examples
        print("\n🎯 Running all examples (use '1d' or '2d' argument to run specific examples)...")
        example_1d_simple()
        example_2d_multiple()


if __name__ == "__main__":
    main()
