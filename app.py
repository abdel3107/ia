from flask import Flask, request, jsonify
import pandas as pd
import joblib

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

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    # Encode symptoms using the MultiLabelBinarizer
    encoded_symptoms = pd.DataFrame(mlb.transform([symptoms]), columns=mlb.classes_)
    encoded_symptoms = encoded_symptoms.reindex(columns=model.feature_names_in_, fill_value=0)



    # Predict the disease
    predicted_disease = model.predict(encoded_symptoms)[0]

    # Get corresponding medications
    prescribed_medications = medications_df.loc[medications_df['Disease'] == predicted_disease, 'Medication']

    if not prescribed_medications.empty:
        medications_list = prescribed_medications.iloc[0]  # Assuming one row per disease
    else:
        medications_list = "No medications found for this disease."

    # Return the prediction and medications
    response = {
        "predicted_disease": predicted_disease,
        "medications": medications_list
    }
    return jsonify(response)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
