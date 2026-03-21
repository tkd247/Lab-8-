# Housing Value Prediction -- Deep Neural Networks

## Project Overview
This project uses a deep neural network (TensorFlow) to predict housing values in Hamilton County, TN based on property features.

This is for **educational purposes only**.

---

## Features Used
- CALC_ACRES (land size)
- YEARBUILT (year constructed)
- SIZEAREA (building area in sq ft)

---

## Model Description
- Neural Network with 4 layers:
  - Input layer
  - 3 hidden layers (128, 64, 32 neurons)
  - Output layer
- Activation: ReLU
- Loss Function: Mean Squared Error (MSE)
- Metric: Mean Absolute Error (MAE)

---

## Why Scaling is Needed
Feature scaling (StandardScaler) ensures all input variables are on the same scale, which improves training stability and model performance.

---

## Results
The model was trained on real housing data and evaluated using MAE.

Predictions are approximate and may not reflect true market values.

---

## Limitations
- Limited number of features
- Does not include location-specific variables
- Sensitive to input ranges
- Not suitable for real property valuation

---

## Ethical Considerations
AI-based predictions can:
- misrepresent property value
- introduce bias
- be misused in financial decisions

This model should NOT be used for:
- appraisals
- lending decisions
- investments

---

## How to Run

### Train Model
```bash
python train_model.py

streamlit run app.py
