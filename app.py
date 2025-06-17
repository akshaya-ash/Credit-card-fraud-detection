import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import joblib

# Load the trained model and scaler
model = joblib.load('model_full.pkl')
scaler = joblib.load('scaler_full.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    # Example data to display for reference
    example_data = {
        'V1': 0.23, 'V2': -0.12, 'V3': 0.01, 'V4': -0.45, 'V5': 0.67,
        'V6': -0.34, 'V7': 0.56, 'V8': -0.23, 'V9': 0.89, 'V10': -0.12,
        'V11': 0.33, 'V12': -0.56, 'V13': 0.23, 'V14': -0.67, 'V15': 0.12,
        'V16': -0.23, 'V17': 0.45, 'V18': -0.33, 'V19': 0.01, 'V20': 0.56,
        'V21': -0.78, 'V22': 0.23, 'V23': 0.45, 'V24': -0.12, 'V25': 0.67,
        'V26': -0.89, 'V27': 0.45, 'V28': -0.12, 'Amount': 500.0
    }
    return render_template('index.html', example_data=example_data)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect values from the form
        features = [float(request.form[f'V{i}']) for i in range(1, 29)]
        amount = float(request.form['Amount'])

        # Scale the amount feature
        amount_scaled = scaler.transform([[amount]])[0][0]
        features.append(amount_scaled)

        # Convert to DataFrame and predict
        input_df = pd.DataFrame([features])
        prediction = model.predict(input_df)[0]
        result = "Fraud ❗" if prediction == 1 else "Not Fraud ✅"

        # Dictionary for displaying values
        features_dict = {f"V{i}": round(features[i-1], 2) for i in range(1, 29)}
        amount_display = round(amount, 2)

        return render_template('index.html',
                               prediction_text=f"Prediction: {result}",
                               features=features_dict,
                               amount=amount_display,
                               example_data=None)
    except Exception as e:
        return render_template('index.html',
                               prediction_text="Error: " + str(e),
                               features=None,
                               amount=None,
                               example_data=None)

if __name__ == "__main__":
    app.run(debug=True)

