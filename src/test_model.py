# test_model.py

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Function to map 'Yes'/'No' to 1/0
def map_yes_no(value):
    if isinstance(value, str):
        return 1 if value == 'Yes' else 0
    return value

# Function to convert Sleep Duration to numeric
def map_sleep_duration(value):
    if value == 'Less than 5 hours':
        return 4
    elif value == '5-6 hours':
        return 5.5
    elif value == '7-8 hours':
        return 7.5
    elif value == '9-10 hours':
        return 9.5
    elif value == 'More than 10 hours':
        return 10
    return np.nan  # For unexpected values

# Load the preprocessed data (same as training)
df = pd.read_csv('data/Student Depression Dataset.csv')  # Make sure this file is preprocessed and clean

# Apply the mapping function to the 'Have you ever had suicidal thoughts?' column
df['Have you ever had suicidal thoughts?'] = df['Have you ever had suicidal thoughts?'].apply(map_yes_no)

# Apply the sleep duration mapping
df['Sleep Duration'] = df['Sleep Duration'].apply(map_sleep_duration)

# Apply the 'Yes'/'No' mapping for other columns where necessary
df['Dietary Habits'] = df['Dietary Habits'].apply(map_yes_no)
df['Study Satisfaction'] = df['Study Satisfaction'].apply(map_yes_no)
df['Job Satisfaction'] = df['Job Satisfaction'].apply(map_yes_no)
df['Family History of Mental Illness'] = df['Family History of Mental Illness'].apply(map_yes_no)

# Features (X) and target (y)
X = df.drop('Depression', axis=1)  # All columns except 'Depression'
y = df['Depression']  # The target column

# Encode binary categorical columns (e.g., 'Gender', 'Have you ever had suicidal thoughts?') using LabelEncoder
label_encoder = LabelEncoder()
X['Gender'] = label_encoder.fit_transform(X['Gender'])  # Encode 'Gender' as 0 or 1
X['Have you ever had suicidal thoughts?'] = label_encoder.fit_transform(X['Have you ever had suicidal thoughts?'])

# Apply OneHotEncoding to multi-class categorical columns (e.g., 'City', 'Profession', 'Degree', etc.)
categorical_columns = ['City', 'Profession', 'Degree']

# Set up OneHotEncoder for multi-class categorical features
one_hot_encoder = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False), categorical_columns)  # Use sparse_output=False for dense output
    ], 
    remainder='passthrough'  # Keep the other columns like 'Age', 'Work Pressure', etc.
)

# Apply OneHotEncoding
X_encoded = one_hot_encoder.fit_transform(X)

# Ensure all columns are numeric before evaluation
X_encoded = np.nan_to_num(X_encoded)  # Convert any NaN values to zero (optional)
X_encoded = X_encoded.astype(np.float64)  # Convert all columns to float64

# Load the trained model
model = joblib.load('model/mental_health_model.pkl')

# Predict on the data (for example, the test data or full dataset)
y_pred = model.predict(X_encoded)

# Evaluate the model
accuracy = accuracy_score(y, y_pred)
conf_matrix = confusion_matrix(y, y_pred)
class_report = classification_report(y, y_pred)

# Display the results
print("Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)



