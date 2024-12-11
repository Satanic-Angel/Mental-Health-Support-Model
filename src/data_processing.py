import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset from CSV
file_path = "data/Student Depression Dataset.csv" # .csv file is our database. 
df = pd.read_csv(file_path)

# Step 1: Handling Missing Data (if applicable)
missing_data = df.isnull().sum()
print("Missing Data per Column:\n", missing_data)

# Step 2: Encoding Categorical Data
# Categorical columns to encode
categorical_cols = ["Gender", "City", "Profession", "Sleep Duration", "Dietary Habits", "Degree", 
                    "Have you ever had suicidal thoughts?", "Family History of Mental Illness"]

label_encoders = {col: LabelEncoder() for col in categorical_cols}

for col in categorical_cols:
    df[col] = label_encoders[col].fit_transform(df[col])

# Step 3: Scaling Numerical Data
numerical_cols = ["Age", "Academic Pressure", "Work Pressure", "CGPA", "Study Satisfaction", 
                  "Job Satisfaction", "Work/Study Hours", "Financial Stress"]

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Step 4: Feature Engineering (optional)
# You can add new features or combine columns for better results (if needed).

# Step 5: Splitting the Dataset into Features and Target
X = df.drop("Depression", axis=1)  # Features
y = df["Depression"]  # Target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Final Preprocessed Data Ready for Model Training
print("Preprocessing complete. Data is ready for training.")

