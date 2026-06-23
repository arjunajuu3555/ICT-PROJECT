
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler # Required for type hints if not directly used

app = Flask(__name__)

# --- Load the trained model and preprocessing objects ---
try:
    model = joblib.load('best_logistic_regression_model.joblib')
    le = joblib.load('label_encoder.joblib')
    scaler = joblib.load('standard_scaler.joblib')
    print("Models and preprocessing objects loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please ensure 'best_logistic_regression_model.joblib', 'label_encoder.joblib', and 'standard_scaler.joblib' are in the same directory as app.py.")
    exit()

# Columns expected by the model (derived from X.columns from training)
# This list must match the feature order used during model training
EXPECTED_COLUMNS = [
    'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x',
    'mag_y', 'mag_z', 'accel_mag', 'accel_x_mean', 'accel_y_mean',
    'accel_z_mean', 'accel_x_std', 'accel_y_std', 'gyro_mag',
    'gyro_x_mean', 'gyro_y_mean', 'gyro_z_mean', 'gyro_x_std',
    'gyro_y_std', 'mag_mag', 'mag_x_std', 'mag_y_std', 'heart_rate',
    'hour_of_day', 'day_of_week'
]

# Numerical columns that were scaled during training
# All features in EXPECTED_COLUMNS were scaled, so this list should be the same.
NUMERICAL_COLS_FOR_SCALING = EXPECTED_COLUMNS

@app.route('/')
def home():
    return render_template('index.html', prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    features = []
    for col in EXPECTED_COLUMNS:
        # Get data from form and convert to float
        try:
            features.append(float(request.form[col]))
        except ValueError:
            return render_template('index.html', prediction_text=f"Error: Invalid input for {col}. Please enter a number.")

    # Convert features to a DataFrame
    input_df = pd.DataFrame([features], columns=EXPECTED_COLUMNS)

    # Apply scaling to numerical features
    input_df[NUMERICAL_COLS_FOR_SCALING] = scaler.transform(input_df[NUMERICAL_COLS_FOR_SCALING])

    # Make prediction
    prediction_encoded = model.predict(input_df)

    # Inverse transform to get original activity name
    prediction_activity = le.inverse_transform(prediction_encoded)[0]

    return render_template('index.html', prediction_text=f"Predicted Activity: {prediction_activity}")

if __name__ == '__main__':
    # Use '0.0.0.0' to make the server accessible from other machines on the network
    # For local development, '127.0.0.1' or 'localhost' is sufficient
    # In Colab, you might use '0.0.0.0' and then expose the port using ngrok or similar if you need external access.
    app.run(host='127.0.0.1', port=5000, debug=True)
