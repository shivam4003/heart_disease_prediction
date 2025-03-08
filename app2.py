import numpy as np
import joblib
from flask import Flask, request, jsonify

# Load the trained model and scaler
model = joblib.load("heart_disease_model.pkl")  # Updated model filename
scaler = joblib.load("scaler.pkl")  # Ensure this file is present

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Heart Disease Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Extract feature values from input
        input_data = np.array(data['features']).reshape(1, -1)

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)

        # Convert prediction to readable format
        result = "The person has heart disease" if prediction[0] == 1 else "The person does not have heart disease"

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)


