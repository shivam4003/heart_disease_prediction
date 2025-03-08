import numpy as np
import joblib
import logging
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Enable CORS for frontend communication

# Initialize Flask app
app = Flask(__name__, template_folder="templates")  # Ensure 'templates' folder exists
CORS(app)  # Enable CORS to allow frontend to interact with API

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the trained model and scaler
try:
    model = joblib.load("heart_disease_model.pkl")  # Ensure this file exists
    scaler = joblib.load("scaler.pkl")  # Ensure this file exists
    logging.info("Model and scaler loaded successfully!")
except Exception as e:
    logging.error(f"Error loading model/scaler: {e}")
    model, scaler = None, None  # Prevent further crashes

@app.route("/")
def home():
    return render_template("index.html")  # Make sure you have an 'index.html' file inside 'templates' folder

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded. Check your files!"}), 500

    try:
        # Get JSON data from request
        data = request.get_json()
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request body"}), 400

        # Convert input to NumPy array
        input_data = np.array(data["features"]).reshape(1, -1)

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)

        # Convert prediction to readable format
        result = "The person has heart disease" if prediction[0] == 1 else "The person does not have heart disease"

        return jsonify({"prediction": result})

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render assigns a dynamic port
    app.run(host="0.0.0.0", port=port, debug=True)



