import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

BASE_DIR = os.path.dirname(__file__)

model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "artifacts/housing_model.h5")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "artifacts/scaler.pkl")
)

st.title("Hamilton County Housing Value Predictor")
st.caption("Educational use only. Predictions are approximate.")

# Inputs
acres = st.number_input("Land area (acres)", 0.01, 20.0, 0.25)

yearbuilt = st.number_input("Year built", 1900, 2026, 2000)

sizearea = st.number_input("Building area (sq ft)", 300, 10000, 1800)

# Prediction
if st.button("Predict"):
    input_df = pd.DataFrame({
        "CALC_ACRES": [acres],
        "YEARBUILT": [yearbuilt],
        "SIZEAREA": [sizearea]
    })

    input_scaled = scaler.transform(input_df)

    prediction = float(model.predict(input_scaled)[0])

    st.success(f"Estimated appraised value: ${prediction:,.0f}")
