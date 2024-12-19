from flask import Flask, request, jsonify
import pandas as pd
import joblib
from models.drugPrescription import recommend_medication


# Initialize Flask app
app = Flask(__name__)

# Load the trained model and MultiLabelBinarizer
model = joblib.load('models/disease_prediction_model.pkl')
mlb = joblib.load('models/mlb.pkl')
medications_df = pd.read_csv('dataset/medications.csv')  # Medications dataset


@app.route('/predict', methods=['POST'])
def predict():
    # Parse JSON input
    data = request.json
    symptoms = data.get('symptoms', [])
    # Ensure that symptoms are provided
    if 'symptoms' not in data:
        return jsonify({"error": "No symptoms provided"}), 400

    symptoms = data['symptoms']

    # Call your recommend_medication function
    medications = recommend_medication(symptoms)

    # Return the predictions as a JSON response
    return jsonify({
        "disease": medications[0],
        "medications": medications[1]
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
