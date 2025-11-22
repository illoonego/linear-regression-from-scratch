"""Basic Linear Regression Example.

This example demonstrates how to use the LinearRegression class
for simple and multiple linear regression tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
from linear_regression.models.linear_regression import LinearRegression
from linear_regression.preprocessing.standart_scaler import StandartScaler
from linear_regression.utils import train_test_split

def example_1d_simple():
    """Example of simple linear regression with 1D data."""
    print("\n1D Simple Linear Regression Example")
    print("-" * 40)

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    X = np.arange(100).reshape(-1, 1) # reshape for 1D feature
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

    # Create and train model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000, fit_intercept=True)
    method = 'gradient_descent'

    print(f"\nTraining model with {method.replace('_', ' ').title()}...\n")
    model.fit(X_train_scaled, y_train, method=method)
    print("\nTraining completed!")

    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Evaluate performance
    r2_train = model.r2_score(y_train, y_pred_train)
    mse_train = model.mean_squared_error(y_train, y_pred_train)

    r2_test = model.r2_score(y_test, y_pred_test)
    mse_test = model.mean_squared_error(y_test, y_pred_test)

    # Unsclae weights to original scale
    weight_unscaled = model.weights_[1] / scaler.std_[0]
    intercept_unscaled = model.weights_[0] - (weight_unscaled * scaler.mean_[0])

    print(f"\nResults:")
    print(f"Training Set - R² Score: {r2_train:.4f}, MSE: {mse_train:.4f}")
    print(f"Testing Set  - R² Score: {r2_test:.4f}, MSE: {mse_test:.4f}")

    print(f"\nComparison of the weights:")
    print(f"True weights: slope={slope}, intercept={intercept}")
    print(f"Learned weights (original scale): slope={weight_unscaled:.2f}, intercept={intercept_unscaled:.2f}")
    print(f"Learning weights (scaled): slope={model.weights_[1]:.2f}, intercept={model.weights_[0]:.2f}")
    print(f"Error (original scale):   slope={abs(slope - weight_unscaled):.2f}, intercept={abs(intercept - intercept_unscaled):.2f}")

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

    # Create and train model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000, fit_intercept=True)
    method = 'gradient_descent'

    print(f"\nTraining model with {method.replace('_', ' ').title()}...\n")
    model.fit(X_train_scaled, y_train, method=method)
    print("\nTraining completed!")

    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Evaluate performance
    r2_train = model.r2_score(y_train, y_pred_train)
    r2_test = model.r2_score(y_test, y_pred_test)
    mse_train = model.mean_squared_error(y_train, y_pred_train)
    mse_test = model.mean_squared_error(y_test, y_pred_test)

    # Unsclae weights to original scale
    weight_unscaled = model.weights_[1] / scaler.std_[0]
    bedroom_unscaled = model.weights_[2] / scaler.std_[1]
    intercept_unscaled = model.weights_[0] - (weight_unscaled * scaler.mean_[0]) - (bedroom_unscaled * scaler.mean_[1])

    print(f"\nResults:")
    print(f"Training Set - R² Score: {r2_train:.4f}, MSE: {mse_train:.4f}")
    print(f"Testing Set  - R² Score: {r2_test:.4f}, MSE: {mse_test:.4f}")
    
    print(f"\nComparison of the weights:")
    print(f"True weights:    size=150.00, bedroom=10000.00, intercept=20000.00")
    print(f"Learned weights (scaled): size={model.weights_[1]:.2f}, bedroom={model.weights_[2]:.2f}, intercept={model.weights_[0]:.2f}")
    print(f"Learned weights (original scale): size={weight_unscaled:.2f}, bedroom={bedroom_unscaled:.2f}, intercept={intercept_unscaled:.2f}\n")
    print(f"Error (original scale):   size={abs(150 - weight_unscaled):.2f}, bedroom={abs(10000 - bedroom_unscaled):.2f}, intercept={abs(20000 - intercept_unscaled):.2f}")

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