import numpy as np
from PIL import Image
import io
import argparse

import uvicorn
from fastapi import FastAPI, Form

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as MSE

# Initialize FastAPI app
app = FastAPI()

# Load the model from the specified path
def load_model(path:str) -> Sequential:
    # Best model
    model = Sequential()
    model.add(Input(shape=(9,)))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(1))
    model.compile(loss=MeanSquaredError(), metrics=[MSE()])
    model.load_weights(path)
    return model

# Predict the price from the input data point
def predict_price(model:Sequential, data_point:list) -> str:
    # Convert the data point to a numpy array and reshape it for the model input
    data_point = np.array(data_point).reshape(1, 9)

    # Use the model to predict the price
    price = model.predict(data_point)

    # Return the predicted price
    return float(price[0, 0])

# Parse the command line arguments and load the model
parser = argparse.ArgumentParser(description="Load a saved TensorFlow model")
parser.add_argument("model_path", help="Path to the saved model")
args = parser.parse_args()
model = load_model(args.model_path)


# Define the API endpoint for predicting the digit from the uploaded image
@app.post("/predict")
async def predict_digit_endpoint(values: str = Form(...)):
    # Split the comma-separated string into a list of floats
    data_point = [float(value) for value in values.split(',')]
    if len(data_point) != 9:
        raise ValueError("Expected 9 comma-separated values")

    # Use the loaded model to predict the price
    predicted_price = predict_price(model, data_point)

    # Return the predicted price as a JSON response
    return {"predicted_price": predicted_price}

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
