import streamlit as st
import numpy as np
import joblib  # For loading the model

# Load trained model
model = joblib.load("titanic_model.pkl")  # Make sure this path is correct!

# Title
st.title("🚢 Titanic Survivor Prediction")

# User Inputs
Pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
Sex = st.radio("Sex", ["Male", "Female"])
Age = st.number_input("Age", min_value=0, max_value=100, value=25)
SibSp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
Parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
Fare = st.number_input("Fare ($)", min_value=0.0, max_value=500.0, value=50.0)
Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encode categorical features
Sex = 1 if Sex == "Male" else 0  # Convert Male=1, Female=0

# Encode 'Embarked'
embarked_mapping = {"C": 0, "Q": 1, "S": 2}  # Adjust based on training data
Embarked = embarked_mapping[Embarked]

# Create input array
input_data = np.array([Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]).reshape(1, -1)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "Survived 🟢" if prediction[0] == 1 else "Did Not Survive 🔴"
    st.subheader(f"Prediction: {result}")
