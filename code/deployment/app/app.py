"""Collect measurements and request predictions from the HTTP API."""
import math
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000").rstrip("/")
CLASSES = {"class_0", "class_1", "class_2"}

st.set_page_config(page_title="Wine Cultivar Predictor", page_icon="🍇")
st.title("Wine Cultivar Predictor")
st.write("Predict one of three Wine dataset cultivar classes from six chemical measurements.")

with st.form("measurements"):
    payload = {
        "alcohol": st.number_input("Alcohol", min_value=0.01, value=13.2, step=0.1),
        "malic_acid": st.number_input("Malic acid", min_value=0.01, value=1.8, step=0.1),
        "magnesium": st.number_input("Magnesium", min_value=0.1, value=100.0, step=1.0),
        "flavanoids": st.number_input("Flavanoids", min_value=0.0, value=2.5, step=0.1),
        "color_intensity": st.number_input("Color intensity", min_value=0.01, value=5.0, step=0.1),
        "proline": st.number_input("Proline", min_value=0.1, value=900.0, step=10.0),
    }
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict) or result.get("prediction") not in CLASSES:
            raise ValueError("Invalid prediction response")
        probabilities = result.get("probabilities")
        if probabilities is not None:
            if not isinstance(probabilities, dict) or set(probabilities) != CLASSES:
                raise ValueError("Invalid class probabilities")
            if not all(isinstance(p, (float, int)) and math.isfinite(p) and 0 <= p <= 1
                       for p in probabilities.values()):
                raise ValueError("Invalid probability values")
            if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5):
                raise ValueError("Probabilities do not sum to one")
        st.success(f"Predicted cultivar: {result['prediction']}")
        if probabilities is not None:
            st.bar_chart({"probability": probabilities})
    except (requests.RequestException, ValueError, TypeError, KeyError):
        st.error("The prediction service is unavailable or returned an invalid response. Please try again shortly.")
