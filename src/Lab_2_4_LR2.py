import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.iter=[]
        self.MSE=[]
        self.hist_coef=[]
        self.hist_intercept=[]
    """
    This next "fit" function is a general function that either calls the *fit_multiple* code that
    you wrote last week, or calls a new method, called *fit_gradient_descent*, not implemented (yet)
    """

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        X_with_bias = np.insert(
            X, 0, 1, axis=1
        )  # Adding a column of ones for intercept

        if method == "least_squares":
            self.fit_multiple(X_with_bias, y)
        elif method == "gradient_descent":
            self.fit_gradient_descent(X_with_bias, y, learning_rate, iterations)

    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # Replace this code with the code you did in the previous laboratory session
                # Add a column of ones to X for the intercept term
        
        # Compute the coefficients using the normal equation
        theta = np.linalg.inv(X.T @ X) @ (X.T @ y)
        
        # Store intercept and coefficients separately
        self.intercept = theta[0]
        self.coefficients = theta[1:]

    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        m = len(y)
    
        # Initialize parameters to small random values.
        # Since X includes a bias column, we only need to initialize coefficients for the other features.
        self.coefficients = np.random.rand(X.shape[1] - 1) * 0.01  # shape: (n_features,)
        self.intercept = np.random.rand() * 0.01

        for epoch in range(iterations):
            # Compute predictions using the current parameters.
            # We assume that the first column of X is the bias (all ones) and is not used here.
            predictions = self.intercept + np.dot(X[:, 1:], self.coefficients)
            
            # Compute error (difference between predictions and true values).
            error = predictions - y

            # Compute the gradient for the intercept (bias) term.
            intercept_gradient = (1 / m) * np.sum(error)
            
            # Compute the gradient for the coefficients (for each feature).
            coefficients_gradient = (1 / m) * np.dot(X[:, 1:].T, error)

            # Update the parameters by taking a step in the opposite direction of the gradient.
            self.intercept -= learning_rate * intercept_gradient
            self.coefficients -= learning_rate * coefficients_gradient

            # Optionally, calculate and print the loss every 10 epochs.
            if epoch % 10 == 0:
                mse = (1 / m) * np.sum(error ** 2)
                self.iter.append(epoch)
                self.MSE.append(mse)
                self.hist_coef.append(self.coefficients)
                self.hist_intercept.append(self.intercept)
                print(f"Epoch {epoch}: MSE = {mse}")

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).
            fit (bool): Flag to indicate if fit was done.

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """

        # Paste your code from last week
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")
        if np.ndim(X) == 1:
            predictions = self.coefficients*X + self.intercept
        else:
            predictions = X @ self.coefficients + self.intercept
        return predictions


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """
    # R^2 Score
    # TODO: Calculate R^2
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r_squared = 1 - (ss_res / ss_tot)
    # Root Mean Squared Error
    # TODO: Calculate RMSE
    rmse = np.sqrt(np.mean((y_true-y_pred)**2))

    # Mean Absolute Error
    # TODO: Calculate MAE
    mae = np.mean(abs(y_true-y_pred))

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    supports string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
    X_transformed = X.copy()

    # Process indices in reverse order to avoid messing up the column positions when inserting new columns.
    for index in sorted(categorical_indices, reverse=True):
        # Extract the categorical column
        categorical_column = X_transformed[:, index]

        # Find the unique categories (works with strings)
        unique_values = np.unique(categorical_column)

        # Create a one-hot encoded matrix using broadcasting
        # Each row gets a 1 in the column corresponding to its category.
        one_hot = (categorical_column[:, None] == unique_values).astype(int)

        # Optionally drop the first level of one-hot encoding
        if drop_first:
            one_hot = one_hot[:, 1:]

        # Delete the original categorical column from X_transformed
        X_transformed = np.delete(X_transformed, index, axis=1)
        X_transformed = np.hstack((X_transformed[:,:index], one_hot,X_transformed[:,index:]))

    return X_transformed



