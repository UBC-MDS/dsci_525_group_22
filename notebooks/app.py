from flask import Flask, request, jsonify
from flask.logging import create_logger
import logging
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
logger = create_logger(app)
logger.setLevel(logging.INFO)

app.run(debug=True)

# 1. Load your model here
model = joblib.load("model.joblib")

# 2. Define a prediction function
def return_prediction(payload):
    # format input_data here so that you can pass it to model.predict()
    df = pd.DataFrame(payload)
    return model.predict(df)


# 3. Set up home page using basic html
@app.route("/")
def index():
    # feel free to customize this if you like
    return """
    <h1>Welcome to our rain prediction service</h1>
    To use this service, make a JSON post request to the /predict url with 25 climate model outputs.
    """


# 4. define a new route which will accept POST requests and return model predictions
@app.route("/predict", methods=["POST"])
def rainfall_prediction():
    content = request.json  # this extracts the JSON content we sent
    logger.info(f"Making prediction for {content}")
    prediction = return_prediction(content)
    prediction = list(
        prediction
    )  # return whatever data you wish, it can be just the prediction
    # or it can be the prediction plus the input data, it's up to you
    logger.info(f"Returning prediction {prediction}")
    return jsonify({"predicion": prediction})
