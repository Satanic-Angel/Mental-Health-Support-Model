# Mental Health Support System
## proceeding with the project 
### a comprehensive guide for my teammates understanding 

Our project is a mental health support system that uses machine learning to predict whether a user is in any form of mental distress or not and also provides remedies for the same, by giving advice and tips. It basically is like a personal therapist for everyone and it will resides locally on their device.

#### The first step:
- We first create a mini-project that predicts whether a user is suffering from depression using a machine learning model trained on the Depression Dataset from Kaggle.
- Which we will further extend by adding advices and personal tips for the model.
- Then we will integrate the model into a chatbot interface using Flask.

## Features
- User-friendly web interface for predictions.
- Machine learning model to classify user input.
- Flask-based backend.
- Deployment on a local server.


## Installation
1. Clone the repository for personal use.
2. Create a python environment either locally in your machine or in a virtual environment:

  ###### for macOS/Linux
You may need to run `sudo apt-get install python3-venv` first on Debian-based OSs then add this line to your vscode terminal:
`python3 -m venv .venv`

###### for Windows
You can also use `py -3 -m venv .venv`
python -m venv .venv

3. Activate the environment by `source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`.
3. Run the Flask app: `python app/app.py`.
4. Access the app at `http://127.0.0.1:5000`.

## File Structure
- `app/`: Contains the backend and frontend files.
- `data/`: Contains the dataset.
- `model/`: Stores the trained model.
- `notebooks/`: Contains the Jupyter Notebook for model training.

## Usage
Input values in the form and click "Predict" to get the result.

## Dataset
- [Depression Dataset on Kaggle](https://www.kaggle.com/)

Here is a comprehensive report detailing all aspects of the project, including the file architecture, backend processes, frontend integration, and dataset analysis:

---

## **1. Project Overview**
The project is a **Mental Health Support System** designed to predict whether an individual is suffering from depression based on various personal and situational factors. It uses machine learning for prediction and a web interface for user interaction.

---

## **2. Dataset Details**
### **Dataset Source:**
- The dataset is sourced from Kaggle and contains **26,000 entries**. 

### **Features:**
1. **ID**: Unique identifier for each record.
2. **Gender**: Categorical (Male/Female/Other).
3. **Age**: Numeric.
4. **City**: Categorical (e.g., urban/rural/metropolitan).
5. **Profession**: Categorical (e.g., student, employed, unemployed).
6. **Academic Pressure**: Numeric (self-reported score).
7. **Work Pressure**: Numeric (self-reported score).
8. **CGPA**: Numeric.
9. **Study Satisfaction**: Numeric.
10. **Job Satisfaction**: Numeric.
11. **Sleep Duration**: Categorical (e.g., '5-6 hours', '7-8 hours').
12. **Dietary Habits**: Categorical (e.g., healthy, unhealthy).
13. **Degree**: Categorical (e.g., Bachelor's, Master's).
14. **Have you ever had suicidal thoughts?**: Categorical (Yes/No).
15. **Work/Study Hours**: Numeric.
16. **Financial Stress**: Numeric.
17. **Family History of Mental Illness**: Categorical (Yes/No).
18. **Depression**: Target variable (1 for depressed, 0 for not depressed).

---

## **3. File Architecture**
### Flask Architecture
```
/your-project-directory
├── /static
│   ├── /css
│   │   └── styles.css
│   ├── /js
│   │   └── scripts.js
│   └── /images
├── /templates
│   └── index.html
├── app.py
├── model
│   └── mental_health_model.pkl
├── predict.py
└── requirements.txt
```

---
### **Django Project Structure**
```
/mental_health_project
├── /mental_health_app
│   ├── /migrations          # Auto-generated database migration files
│   ├── /static              # Static files (CSS, JS, images)
│   │   ├── /css
│   │   │   └── styles.css   # Custom styles for the app
│   │   ├── /js
│   │   │   └── scripts.js   # Frontend interaction logic
│   │   └── /images          # Any static images used in the app
│   ├── /templates
│   │   └── index.html       # The main frontend interface
│   ├── admin.py             # Django admin configuration
│   ├── apps.py              # App-specific configuration
│   ├── models.py            # Database models (if needed)
│   ├── tests.py             # Unit tests for the app
│   ├── views.py             # Backend logic for handling requests
│   └── urls.py              # URL routing for the app
├── /model
│   ├── mental_health_model.pkl        # Trained machine learning model
│   ├── label_encoder_gender.pkl       # Encoder for 'Gender' column
│   ├── label_encoder_city.pkl         # Encoder for 'City' column
│   └── (scaler.pkl - optional)        # Scaler for numerical columns (if used)
├── manage.py                          # Django's command-line utility
└── /mental_health_project
    ├── settings.py                    # Project settings
    ├── urls.py                        # Project-wide URL configuration
    ├── wsgi.py                        # WSGI entry point for deployment
    └── asgi.py                        # ASGI entry point for asynchronous deployment
```

---

## **4. Backend Process**
### **4.1 Model Training**
- **Libraries Used**: Scikit-learn, pandas, NumPy.
- **Steps:**
  - Loaded the dataset.
  - Handled missing values (e.g., imputing or removing rows).
  - Encoded categorical columns using **LabelEncoder**.
  - Scaled numerical columns using **StandardScaler**.
  - Trained a **Random Forest Classifier** to predict the target variable (`Depression`).
  - Saved the trained model and encoders as `.pkl` files using **joblib**.

---

### **4.2 Django Backend**
1. **Views**:
   - `index`: Renders the main web interface (`index.html`).
   - `predict`: Handles POST requests to receive input data, preprocess it, and return predictions.

2. **Prediction Logic**:
   - Extracted input from the form submitted via the frontend.
   - Encoded categorical features using the saved encoders.
   - Scaled numerical features (if scaling was applied during training).
   - Predicted the output using the trained model.
   - Returned the result (1 for depressed, 0 for not depressed).

---

## **5. Frontend Integration**
### **5.1 HTML (index.html)**
- Designed a form-based interface to collect user inputs for all dataset parameters.
- Used responsive design techniques for compatibility with various devices.

### **5.2 CSS (styles.css)**
- Added styles to improve the aesthetics of the form and layout.

### **5.3 JavaScript (scripts.js)**
- Handled client-side validation for form inputs to ensure data integrity before submission.

### **5.4 Interaction with Backend**
- The form sends data via POST requests to the Django `predict` endpoint.
- The result (prediction) is displayed dynamically on the page after submission.

---

## **6. Dataset Preprocessing for Deployment**
- Encoded features (`Gender`, `City`, etc.) using pre-trained **LabelEncoders**.
- Applied numerical transformations for categorical values (e.g., Sleep Duration mapping).
- Saved all preprocessing logic for reuse in the prediction phase.

---

## **7. Improvements**
1. **Model Accuracy**: 
   - Improved from initial values to a decent value of ~78% without using strategies like oversampling and hyperparameter tuning.
2. **Scalability**:
   - Model is pretty scalable using flask but it can be scaled to a large number of users.
   - Transitioned from Flask to Django to leverage Django's robust features and scalability.
3. **Separation of Concerns**:
   - Ensured separate files for frontend (`HTML/CSS/JS`) and backend (`Flask/Django views`).
4. **Future Enhancements**:
   - Add a database to store user inputs and predictions for analysis.
   - Implement user authentication for personalized experiences.

---

## **8. Testing**
1. **Local Testing**:
   - Tested predictions using cURL, Python requests, and manual form submissions.
   - results are accepted in a json file format.
2. **Endpoints**:
   - `GET /`: Loads the homepage.
   - `POST /predict`: Returns the depression prediction based on the provided inputs.

---

## **9. Summary**
This project successfully integrates machine learning and web development to build an end-to-end Mental Health Support System. It includes:
- A **robust machine learning model** trained on a real-world dataset.
- A **user-friendly web interface** developed using Django, with clear separation of frontend and backend components.
- **Reusable components** for encoding and scaling to ensure the consistency of predictions.

The project is now ready for deployment, and further enhancements can be implemented for real-world use cases. Let me know if you need assistance with any additional features!