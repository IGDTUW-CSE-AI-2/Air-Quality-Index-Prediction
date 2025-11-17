import streamlit as st
import pickle
import numpy as np

# Load model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

st.title("AQI Classification App")
st.write("Enter feature values below to get a prediction.")

feature1 = st.number_input("Sulfur dioxide (SO2)")
feature2 = st.number_input("Nitrogen Dioxide (NO2)")
feature3 = st.number_input("Respirable Suspended Particulate Matter (RSPM)")
feature4 = st.number_input("Suspended Particulate Matter (SPM)")

input_data = np.array([[feature1, feature2, feature3, feature4]])

if st.button("Predict"):
    pred_encoded = model.predict(input_data)
    prediction = le.inverse_transform(pred_encoded)[0]
    st.success(f"Predicted Class: **{prediction}**")