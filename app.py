import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

st.title("Crop Recommendation System 🌱")

@st.cache_resource
def load_model_scaler():
    try:
        if os.path.exists("model.pkl") and os.path.exists("scaler.pkl"):
            with open("model.pkl", "rb") as f:
                model = pickle.load(f)
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            return model, scaler
        else:
            # Fallback: train simple model from csv
            df = pd.read_csv("Crop_recommendationV2.csv")
            X = df.drop(columns=['label'])
            y = df['label']
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
            model.fit(X_train_scaled, y_train)
            return model, scaler
    except Exception as e:
        st.error(f"Model load error: {e}. Using fallback.")
        class DummyModel:
            def predict(self, X):
                return ['Rice']
        class DummyScaler:
            def transform(self, X):
                return X
        return DummyModel(), DummyScaler()

st.write("Enter soil and environmental conditions")

model, scaler = load_model_scaler()

N = st.number_input("Nitrogen")
P = st.number_input("Phosphorus")
K = st.number_input("Potassium")
temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
ph = st.number_input("pH")
rainfall = st.number_input("Rainfall")
soil_moisture = st.number_input("Soil Moisture")
soil_type = st.number_input("Soil Type")
sunlight_exposure = st.number_input("Sunlight Exposure")
wind_speed = st.number_input("Wind Speed")
co2_concentration = st.number_input("CO2 Concentration")
organic_matter = st.number_input("Organic Matter")
irrigation_frequency = st.number_input("Irrigation Frequency")
crop_density = st.number_input("Crop Density")
pest_pressure = st.number_input("Pest Pressure")
fertilizer_usage = st.number_input("Fertilizer Usage")
growth_stage = st.number_input("Growth Stage")
urban_area_proximity = st.number_input("Urban Area Proximity")
water_source_type = st.number_input("Water Source Type")
frost_risk = st.number_input("Frost Risk")
water_usage_efficiency = st.number_input("Water Usage Efficiency")

if st.button("Predict Crop"):

    data = np.array([[N,P,K,temperature,humidity,ph,rainfall,
                      soil_moisture,soil_type,sunlight_exposure,
                      wind_speed,co2_concentration,organic_matter,
                      irrigation_frequency,crop_density,pest_pressure,
                      fertilizer_usage,growth_stage,urban_area_proximity,
                      water_source_type,frost_risk,water_usage_efficiency]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    st.success(f"Recommended Crop: {prediction[0]}")