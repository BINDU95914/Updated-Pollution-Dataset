import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the classifier
with open('updated_Air pollution set.pkl', 'rb') as file:
    classifier = pickle.load(file)

# Load or initialize the scaler (if required by your model)
try:
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)  # Load a pre-fitted scaler
except FileNotFoundError:
    scaler = None  # Assume no scaling if scaler is not available

# App title
st.title("Air Quality Prediction App")

# Input fields with consistent data types
PM2_5 = st.number_input("Enter PM2.5 level (µg/m³):", min_value=0.0, value=0.0, step=0.1, format="%.2f")
PM10 = st.number_input("Enter PM10 level (µg/m³):", min_value=0.0, value=0.0, step=0.1, format="%.2f")
NO2 = st.number_input("Enter NO2 level (µg/m³):", min_value=0.0, value=0.0, step=0.1, format="%.2f")
SO2 = st.number_input("Enter SO2 level (µg/m³):", min_value=0.0, value=0.0, step=0.1, format="%.2f")
Proximity_to_Industrial_Areas = st.number_input("Enter proximity to industrial areas (km):", min_value=0.0, value=0.0, step=0.1, format="%.2f")
Population_Density = st.number_input("Enter population density (people/sq km):", min_value=0, value=0, step=1)

# Predict button
if st.button("Predict"):
    # Prepare input data as a 2D array
    input_data = np.array([[PM2_5, PM10, NO2, SO2, Proximity_to_Industrial_Areas, Population_Density]])
    
    # Scale the input data if a scaler is available
    if scaler:
        try:
            encoded_input = scaler.transform(input_data)  # Scale inputs
        except Exception as e:
            st.error(f"Scaler transformation error: {e}")
            encoded_input = input_data  # Use raw input as fallback
    else:
        encoded_input = input_data  # Use raw input if no scaler is available

    # Make prediction
    try:
        prediction = classifier.predict(encoded_input)
        st.success(f"The predicted air quality is: {prediction[0]}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
