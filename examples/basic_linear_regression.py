"""Basic Linear Regression Example.

This example demonstrates how to use the LinearRegression class
for simple and multiple linear regression tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
from linear_regression.models.linear_regression import LinearRegression

def example_1d_simple():
    """Example of simple linear regression with 1D data."""
    print("\n1D Simple Linear Regression Example")
    print("-" * 40)

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    X = np.array([[1], [2], [3], [4], [5]])
    slope = 2.0
    intercept = 3.0
    y_true = slope * X.flatten() + intercept + np.random.randn(5)  # y = 2x + 3 + noise
    print(f"Data points: {len(X)}")
    print(f"True weights: slope={slope}, intercept={intercept}")

    # Create and train model
    model = LinearRegression(learning_rate=0.01, n_iterations=2500, fit_intercept=True)
    method = 'gradient_descent'

    print(f"\nTraining model with {method.replace('_', ' ').title()}...\n")
    model.fit(X, y_true, method=method)
    print("\nTraining completed!")

    # Make predictions and evaluate results
    y_pred = model.predict(X)
    r2 = model.r2_score(y_true, y_pred)
    mse = model.mean_squared_error(y_true, y_pred)

    print(f"\nResults:")
    print(f"Learned weights: slope={model.weights_[1]:.2f}, intercept={model.weights_[0]:.2f}")
    print(f"R² Score:        {r2:.4f}")
    print(f"MSE:             {mse:.4f}")

def example_2d_multiple():
    """Example of multiple linear regression with 2D data."""
    print("\n2D Multiple Linear Regression Example")
    print("-" * 40)

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    np.random.seed(42)
    size_sqft = np.random.uniform(800, 2500, 100)  # Real house sizes
    bedrooms = np.random.randint(1, 5, 100)        # Real bedroom counts
    X = np.column_stack((size_sqft, bedrooms))
    noise = np.random.randn(100) * 10000
    price = 150 * size_sqft + 10000 * bedrooms + 20000 + noise  # price = 150*size + 10000*bedrooms + 20000 + noise
    print(f"Data points: {len(X)}")
    print("True weights: size coefficient=150, bedroom coefficient=10000, intercept=20000")

    # Create and train model
    model = LinearRegression(learning_rate=0.0000001, n_iterations=5000, fit_intercept=True)
    method = 'gradient_descent'

    print(f"\nTraining model with {method.replace('_', ' ').title()}...\n")
    model.fit(X, price, method=method)
    print("\nTraining completed!")

    # Make predictions and evaluate results
    price_pred = model.predict(X)
    r2 = model.r2_score(price, price_pred)
    mse = model.mean_squared_error(price, price_pred)

    print(f"\nResults:")
    print(f"Learned weights: size coefficient={model.weights_[1]:.2f}, bedroom coefficient={model.weights_[2]:.2f}, intercept={model.weights_[0]:.2f}")
    print(f"R² Score:        {r2:.4f}")
    print(f"MSE:             {mse:.4f}")
    
    print(f"\nComparison with True Values:")
    print(f"True:    size=150.00, bedroom=10000.00, intercept=20000.00")
    print(f"Learned: size={model.weights_[1]:.2f}, bedroom={model.weights_[2]:.2f}, intercept={model.weights_[0]:.2f}")
    print(f"Error:   size={abs(150 - model.weights_[1]):.2f}, bedroom={abs(10000 - model.weights_[2]):.2f}, intercept={abs(20000 - model.weights_[0]):.2f}")

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