import pickle
import numpy as np

# Load the trained model and scalers (or encoders) that were used during training
model = pickle.load(open('model/mental_health_model.pkl', 'rb'))
label_encoder_gender = pickle.load(open('model/label_encoder_gender.pkl', 'rb'))  # Example if gender was encoded
scaler = pickle.load(open('model/scaler.pkl', 'rb'))  # Example if scaling was applied to numeric features

def preprocess_input(gender, age, city, profession):
    # Preprocessing steps: Encode categorical features and scale numerical features
    if gender == 'Male':
        gender = 1
    else:
        gender = 0

    # You can include any other preprocessing steps you did during training here

    # Prepare input data as a numpy array (make sure to scale/encode the features correctly)
    input_data = np.array([[gender, age]])

    # Apply scaling if necessary
    input_data = scaler.transform(input_data)

    return input_data

def predict(gender, age, city, profession):
    # Preprocess the input
    input_data = preprocess_input(gender, age, city, profession)
    
    # Make a prediction using the model
    prediction = model.predict(input_data)
    
    # Return prediction result: 'Depressed' or 'Not Depressed'
    if prediction == 1:
        return 'Depressed'
    else:
        return 'Not Depressed'

# Example usage
if __name__ == "__main__":
    gender = 'Female'
    age = 25
    city = 'Bangalore'
    profession = 'Student'

    result = predict(gender, age, city, profession)
    print(f"Prediction: {result}")
