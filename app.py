import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load artifacts
model = joblib.load("artifacts/model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

st.title("Hamilton County Housing Value Predictor")
st.caption("Educational use only. Predictions are approximate.")

# Inputs
acres = st.number_input(
    "Land area (acres)",
    min_value=0.01,
    max_value=20.0,
    value=0.25,
    step=0.01
)

land_value = st.number_input(
    "Land value ($)",
    min_value=0,
    max_value=1000000,
    value=50000
)

build_value = st.number_input(
    "Building value ($)",
    min_value=0,
    max_value=1000000,
    value=150000
)

# Prediction
if st.button("Predict"):

    input_df = pd.DataFrame({
        "CALC_ACRES": [acres],
        "LAND_VALUE": [land_value],
        "BUILD_VALUE": [build_value]
    })

    input_scaled = scaler.transform(input_df)

    prediction = float(model.predict(input_scaled)[0])

    st.success(f"Estimated appraised value: ${prediction:,.0f}")
