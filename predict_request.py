import requests
import json

url = 'http://127.0.0.1:5000/predict'  # Flask server URL

# Example data to send for prediction
data = {
    'Gender': 'Male',
    'Age': 30,
    'City': 'Visakhapatnam',
    'Profession': 'Student',
    'Academic Pressure': 4.0,
    'Work Pressure': 3.0,
    'CGPA': 8.5,
    'Study Satisfaction': 4.0,
    'Job Satisfaction': 3.0,
    'Sleep Duration': '5-6 hours',
    'Dietary Habits': 'Healthy',
    'Degree': 'B.Tech',
    'Have you ever had suicidal thoughts?': 'No',
    'Work/Study Hours': 6.0,
    'Financial Stress': 2.0,
    'Family History of Mental Illness': 'No'
}

# Send a POST request to Flask with the correct Content-Type
response = requests.post(url, json=data)  # Send as JSON

# Print the response
print(response.json())
