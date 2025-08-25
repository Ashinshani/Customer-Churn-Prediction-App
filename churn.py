# Gender = 1 - Female, 0 - Male
# Churn = 1 - Yes, 0 - No
# scaler scaled and saved as scaler.pkl

import streamlit as st
import joblib
import numpy as np
from PIL import Image

st.markdown("""
<style>
    .stButton > button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #FF0000;
        color: white;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

scaler = joblib.load("scaler.pkl")
model = joblib.load("best_model.pkl")   

st.title(" Customer Churn Prediction App 🤖")
# From local file
st.image("churn.png", width=650)

st.divider()

st.write("Please provide the following details to predict if a customer will churn or not:")
st.divider()

age = st.number_input("Enter Age", min_value=10, max_value=100, value=30)
Tenure = st.number_input("Enter Tenure (in months)", min_value=0, max_value=130, value=10)
monthlyCharges = st.number_input("Enter Monthly Charges ", min_value=0, max_value=200, value=70)
gender = st.selectbox("Enter the Gender", ["Male", "Female"])

st.divider()

predictbutton = st.button("Predict")

st.divider()

if predictbutton:
    gender_selected = 1 if gender == "Female" else 0
    x = [age, Tenure, monthlyCharges, gender_selected]
    x1 = np.array(x)
    x_array = scaler.transform([x1])
    prediction = model.predict(x_array)[0]
    predicted = "Churn 😢🚪🏃‍♂️" if prediction == 1 else "Not Churn 😀📱💳"
    st.write(f"Predicted result: {predicted}")
    st.balloons()

else:
    st.write("Click on Predict button to get the result")