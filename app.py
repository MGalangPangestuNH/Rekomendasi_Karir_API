from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load or recreate the scaler and label encoder (must match training)
# Assuming you saved the scaler and label encoder during training
# If not, we'll recreate them with the same configuration
features = [
    'Linguistic', 'Musical', 'Bodily', 'Logical - Mathematical', 'Spatial-Visualization',
    'Interpersonal', 'Naturalist', 'math_score', 'physics_score', 'biology_score',
    'english_score', 'history_score', 'chemistry_score', 'geography_score',
    'weekly_self_study_hours', 'absence_days'
]

# For this example, we'll initialize a new scaler and label encoder
# In production, you should save and load these from training
scaler = StandardScaler()
label_encoder = LabelEncoder()

# Load example data to fit scaler and encoder (replace with your actual training data loading)
# This is a placeholder; ideally, load your original training data
df1 = pd.read_csv("Dataset1.csv")
df2 = pd.read_csv("Dataset2.csv")
df = pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)
df['Job profession'] = df['Job profession'].str.strip()
X = df[features].astype(float).fillna(0).replace([np.inf, -np.inf], 0)
y = df['Job profession']
scaler.fit(X)
label_encoder.fit(y)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Validate input features
        if not all(feature in data for feature in features):
            return jsonify({'error': 'Missing some input features'}), 400

        # Create input DataFrame
        input_data = pd.DataFrame([data], columns=features)

        # Preprocess input
        input_data = input_data.astype(float).fillna(0).replace([np.inf, -np.inf], 0)
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_job = label_encoder.inverse_transform(predicted_class)[0]

        return jsonify({
            'predicted_job': predicted_job,
            'confidence': float(np.max(prediction))
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.getenv('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)