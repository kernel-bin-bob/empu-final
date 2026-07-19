from __future__ import annotations
import streamlit as st
from joblib import load
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

def format_float(num: float, decimal_precision: int = 2):
    fmt_string = f"%.{decimal_precision}f"
    return fmt_string % num

def get_model_name(model_index: int) -> str:
    model_names = ["Model #1 (Small train dataset)", "Model #2 (Medium train dataset)", "Model #3 (Large train dataset)"]
    return model_names[model_index]

def display_model_info(model_info_path: str):
    model_info = load(model_info_path)
    st.markdown(f"""**Model info and evaluation:**\n
- Train dataset size: {model_info['train_dataset_size']}\n
- Test dataset size: {model_info['test_dataset_size']}\n
- R-squared: {format_float(model_info['r_squared'], decimal_precision=4)}\n
- Mean square error: {format_float(model_info['mse'])}\n
- Root Mean Square Error: {format_float(model_info['rmse'])}\n
- Mean Absolute Error: {format_float(model_info['mae'])}"""
    )

def load_and_predict(X: ArrayLike, filename: str) -> ArrayLike:
    # Deserialize and load the regression model and use it to predict on user provided data.

    # This function takes a file name "filename" that has a default value.
    # It uses Joblib "load" to load the model using the provided file name.
    # When the model is loaded, call its `predict` method on provied data.

    # Args:
    #     X (array-like): User provided data used for prediction.
    #     filename (str): Name of the file that is used to store the model.

    # Returns:
    #     np.ndarray: Predicted value.
    
    model = load(filename)
    y = model.predict(X)
    return y

def create_streamlit_app():
    # Creates a Streamlit web application for making predictions with a simple regression model.

    # This function sets up a Streamlit app with a user interface for inputting a single feature 
    # value and making predictions using a pre-trained regression model. The app includes:
    
    # - A title displayed at the top of the app.
    # - A slider for the user to select an input feature value within a specified range (-3.0 to 3.0).
    # - A "Predict value" button that, when clicked, triggers the prediction process.
    # - Upon clicking the "Predict value" button, the function:
    #     - Calls `load_and_predict`, passing the selected feature as input, to load the regression model 
    #       and make a prediction.
    #     - Displays the prediction result on the app.
    #     - Calls `visualize_difference`, passing the input feature and the prediction result, 
    #       to visualize the difference between the predicted value and the actual value in the original dataset.

    # Note: This function does not return any value. It directly manipulates the Streamlit app"s UI by 
    # writing content and rendering UI elements.

    st.header(f"Simple regression model prediction")

    # Feature value slider
    sliderConainer = st.container(border=True)
    X = 1.27
    with sliderConainer:
        X = st.slider("Input feature for prediction", -3.0, 3.0, X)

    # Model dropdown
    model_indices = range(3)
    model_index = 0

    dropdownConainer = st.container(border=True)
    with dropdownConainer:
        model_index = st.selectbox(
            "Select the model to make a prediction",
            model_indices,
            format_func=get_model_name
        )

    user_readable_model_index = model_index + 1
    model_dir_path = f"models/{user_readable_model_index}"

    # Display model info
    with dropdownConainer:
        model_info_path = f"{model_dir_path}/info.joblib"
        display_model_info(model_info_path)

    if st.button("Predict", type="primary"):
        model_filename = f"{model_dir_path}/model.joblib"
        y = load_and_predict([[X]], filename=model_filename)

        X_filename = f"dataset/X.joblib"
        y_filename = f"dataset/y.joblib"
        visualize_difference(X, y, X_filename, y_filename)

def visualize_difference(input_feature: float, prediction: ArrayLike, X_filename: str, y_filename: str):
    # Deserialize and load the initial datasets. Calculate the difference between actual data
    # in the "y" dataset and the predicted value for a given "input_feature".

    # Visualize the difference by plotting the entire "X" & "y" as a Scatter plot. Then add
    # a blue dot that represents the actual target value, and a red dot that represents the predicted target value for the given "input_feature".
    # Add a dashed line connects these points, highlighting the difference between them, which is annotated on the plot.

    # Args:
    #     input_feature (float): User provided data used for prediction.
    #     prediction (array-like): Predicted value.
    #     X_filename (array-like): Test feature matrix filename.
    #     y_filename (array-like): Validation target values filename.

    X = load(X_filename)
    y = load(y_filename)

    actual_target = y[_index_of_closest(X, input_feature)]
    prediction_value = prediction[0]
    difference = format_float(abs(actual_target - prediction_value))

    fig = plt.figure(figsize=(6,4))
    plt.scatter(X, y, color="grey")
    plt.scatter(input_feature, actual_target, color="blue")
    plt.scatter(input_feature, prediction_value, color="red")
    plt.legend(["Dataset", "Actual target", "Predicted target"])
    plt.title("Prediction vs actual target")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.grid()
    plt.plot([input_feature, input_feature], [actual_target, prediction_value], "k--")
    plt.annotate(f"  Difference ({difference})", [input_feature, (actual_target + prediction_value) / 2])

    resultsContainer = st.container(border=True)
    with resultsContainer:
        st.markdown(f"""**Prediction results:**\n
- Input feature: {format_float(input_feature)}\n
- Actual target: {format_float(actual_target)}\n
- Predicted result: {format_float(prediction_value)}\n
- Difference: {difference}"""
        )
    st.pyplot(fig)

# This is a helper function. No need to edit it
def _index_of_closest(X: ArrayLike, k: float) -> int:
    # This function takes an array-like object `X` and a float `k`, and returns the index of the 
    # element in `X` that is closest to `k`. The function first converts `X` into a NumPy array 
    # (if it isn"t one already) to ensure compatibility with NumPy operations. It then calculates 
    # the absolute difference between each element in `X` and `k`, identifies the minimum value 
    # among these differences, and returns the index of this minimum difference.

    # Args:
    #     X (ArrayLike): An array-like object containing numerical data. It can be a list, tuple, 
    #   or any object that can be converted to a NumPy array.
    #     k (float): The target value to which the closest element in `X` is sought.

    # Returns:
    #     int: The index of the element in `X` that is closest to the value `k`.
    # Returns:
    #     int: Index for the closest value to k in X.
    # Finds the index of the element in `X` that is closest to the value `k`.

    X = np.asarray(X)
    idx = (np.abs(X - k)).argmin()
    return idx


if __name__ == "__main__":
    create_streamlit_app()