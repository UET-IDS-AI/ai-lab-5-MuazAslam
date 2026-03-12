"""
AIstats_lab.py

Student starter file for the Regularization & Overfitting lab.
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# =========================
# Helper Functions
# =========================

def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# =========================
# Q1 Lasso Regression
# =========================

def lasso_regression_diabetes(lambda_reg=0.1, lr=0.01, epochs=2000):
    """
    Implement Lasso regression using gradient descent.
    """

    # TODO: Load diabetes dataset
    data = load_diabetes()
    X = data.data
    y = data.target
    # TODO: Train/test split
    X_train , X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)
    # TODO: Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # TODO: Add bias column
    X_train = add_bias(X_train)
    X_test = add_bias(X_test)
    # TODO: Initialize theta
    theta = np.zeros(X_train.shape[1])
    m = len(y_train)
    # TODO: Implement gradient descent with L1 regularization
    for _ in range(epochs):
        predictions = X_train @ theta
        error = predictions - y_train

        gradient = (1/m) * (X_train.T @ error)

         # L1 regularization (except bias)
        gradient[1:] += lambda_reg * np.sign(theta[1:])

        theta -= lr * gradient

    # TODO: Compute predictions
    train_pred = X_train @ theta
    test_pred = X_test @ theta
    # TODO: Compute metrics

    train_mse = mse(y_train, train_pred)
    test_mse = mse(y_test, test_pred)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    return train_mse, test_mse, train_r2, test_r2, theta

    raise NotImplementedError


# =========================
# Q2 Polynomial Overfitting
# =========================

def polynomial_overfitting_experiment(max_degree=10):
    """
    Study overfitting using polynomial regression.
    """

    # TODO: Load dataset
    data = load_diabetes()
    X = data.data
    y = data.target
    # TODO: Select BMI feature only
    X = X[:, 2].reshape(-1, 1)
    # TODO: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    degrees = []
    train_errors = []
    test_errors = []

    # TODO: Loop through polynomial degrees
    for d in range(1, max_degree + 1):

        degrees.append(d)
        # TODO: Create polynomial features
        poly = PolynomialFeatures(degree=d)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # TODO: Fit regression using normal equation

        theta = np.linalg.pinv(X_train_poly.T @ X_train_poly) @ X_train_poly.T @ y_train

        # TODO: Compute train/test errors
        train_pred = X_train_poly @ theta
        test_pred = X_test_poly @ theta

        train_mse = mse(y_train, train_pred)
        test_mse = mse(y_test, test_pred)

        train_errors.append(train_mse)
        test_errors.append(test_mse)

        return {
            "degrees": degrees,
            "train_mse": train_errors,
            "test_mse": test_errors
            }
    

    raise NotImplementedError
