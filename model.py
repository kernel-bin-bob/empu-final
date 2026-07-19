from __future__ import annotations
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
import os
from pathlib import Path
from numpy.typing import ArrayLike
from joblib import dump


def train_regression_model(X_train: ArrayLike, y_train: ArrayLike) -> LinearRegression:
    # Train a regression model using the provided training data.

    # This function takes training data consisting of a feature matrix "X_train" and
    # corresponding target values "y_train", and trains a linear regression model.
    # The trained model is returned for further use.

    # Args:
    #     X_train (array-like): Training feature matrix.
    #     y_train (array-like): Target values for training.

    # Returns:
    #     sklearn.linear_model.LinearRegression: Trained linear regression model.

    return LinearRegression().fit(X_train, y_train)

def evaluate_regression_model(model: LinearRegression, X_test: ArrayLike, y_test: ArrayLike):
    # Evaluate the performance of a regression model on test data.

    # This function takes a trained regression "model", test feature matrix "X_test",
    # and corresponding test target values "y_test". It calculates Mean Squared Error (MSE)
    # and prints it in terminal.

    # Args:
    #     model (sklearn.linear_model.LinearRegression): Trained regression model to be evaluated.
    #     X_test (array-like): Test feature matrix.
    #     y_test (array-like): Validation target values.
    
    # Returns:
    #     dict: A dictionary containing performance metrics:
    #           - "MSE": Mean Squared Error.
    #           - "RMSE": Root Mean Squared Error.
    #           - "MAE": Mean Absolute Error.
    #           - "R-squared": R-squared score.
    
    y_predicted = model.predict(X_test)

    mse = mean_squared_error(y_test, y_predicted)
    rmse = root_mean_squared_error(y_test, y_predicted)
    mae = mean_absolute_error(y_test, y_predicted)
    r_squared = r2_score(y_test, y_predicted)

    return {
      "mse": mse,
      "rmse": rmse,
      "mae": mae,
      "r_squared": r_squared,
    }

if __name__ == "__main__":
    dataset_size = 10000
    test_size = 0.2

    # Define model datasets size in percent
    dataset_train_sizes = [0.005, 0.02, 1]

    # Create dataset
    X, y = make_regression(n_samples=dataset_size, n_features=1, noise=20, random_state=42)
    X = np.interp(X, (X.min(), X.max()), (-3, 3))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for iteration in range(len(dataset_train_sizes)):
        user_readable_iteration = iteration + 1
        
        # Prepare training data
        X_train_current, y_train_current = shuffle(
            X_train, 
            y_train, 
            n_samples=int(len(X_train) * dataset_train_sizes[iteration]),
            random_state=42
        )

        print(f"Training model #{user_readable_iteration}")
        print(f"Training set size: {len(X_train_current)}")

        # Train and evaluate the model
        model = train_regression_model(X_train_current, y_train_current)
        eval_metrcis = evaluate_regression_model(model, X_test, y_test)
        model_info = eval_metrcis
        model_info["train_dataset_size"] = len(X_train_current)
        model_info["test_dataset_size"] = len(X_test)
        print(f"Model #{user_readable_iteration}, Evaluation metrics: {model_info}")

        # Create directory for model and datasets
        dirname = os.path.dirname(__file__)
        model_dir_path = f"{dirname}/models/{user_readable_iteration}"
        Path(model_dir_path).mkdir(parents=True, exist_ok=True)

        dataset_dir_path = f"{dirname}/dataset"
        Path(dataset_dir_path).mkdir(parents=True, exist_ok=True)

        # Save model with metadata and datasets
        X_filename = f"{dirname}/dataset/X.joblib"
        y_filename = f"{dirname}/dataset/y.joblib"
        dump(X, X_filename)
        dump(y, y_filename)

        model_filename = f"{model_dir_path}/model.joblib"
        dump(model, model_filename)

        info_filename = f"{model_dir_path}/info.joblib"
        dump(eval_metrcis, info_filename)