import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('model/mental_health_model.pkl')

# Load the encoder and scaler (if you used them during training)
label_encoder_gender = joblib.load('model/label_encoder_gender.pkl')
label_encoder_city = joblib.load('model/label_encoder_city.pkl')
# scaler = joblib.load('model/scaler.pkl')

# Route for the homepage (Optional)
@app.route('/')
def home():
    return render_template('index.html')

# Function to preprocess input features (similar to training data preprocessing)
def preprocess_features(input_data):
    # Example of encoding categorical columns with LabelEncoder
    input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])
    input_data['City'] = label_encoder_city.transform(input_data['City'])
    
    # Example: Converting 'Sleep Duration' (e.g., '5-6 hours') to numerical values
    input_data['Sleep Duration'] = input_data['Sleep Duration'].apply(lambda x: {'Less than 5 hours': 1, '5-6 hours': 2, '7-8 hours': 3, 'More than 8 hours': 4}.get(x, 0))

    # Convert 'Have you ever had suicidal thoughts?' to 1/0
    input_data['Have you ever had suicidal thoughts?'] = input_data['Have you ever had suicidal thoughts?'].map({'Yes': 1, 'No': 0})

    # Scaling numerical columns (if you used scaling during training)
    numerical_columns = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction', 'Financial Stress', 'Work/Study Hours', 'Family History of Mental Illness']
    input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])

    return input_data

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.get_json()  # Data sent as JSON

    # Convert the input data to a dataframe
    input_data = pd.DataFrame([data])

    # Ensure the model receives the correct features
    X_new = input_data[['Gender', 'Age', 'City', 'Profession', 'Academic Pressure', 
                        'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction',
                        'Sleep Duration', 'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts?',
                        'Work/Study Hours', 'Financial Stress', 'Family History of Mental Illness']]

    # Perform necessary preprocessing (encode categorical features, scale, etc.)
    X_new_encoded = preprocess_features(X_new)  # Apply preprocessing steps

    # Make a prediction
    prediction = model.predict(X_new_encoded)

    # Return the prediction as a JSON response
    return jsonify({'prediction': int(prediction[0])})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
