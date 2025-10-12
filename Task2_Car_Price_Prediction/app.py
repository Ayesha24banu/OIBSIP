# =========================================
# app.py - Car Price Prediction (Streamlit)
# =========================================

import os
import pickle
import logging
import pandas as pd
import streamlit as st
from src.predict import preprocess_input, predict_price

# -------------------------------
# Directories & Logging
# -------------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "app.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# -------------------------------
# Paths for model and encoders
# -------------------------------
MODEL_PATH = "models/model.pkl"
ENCODERS_PATH = "models/encoders.pkl"

# -------------------------------
# Load Model
# -------------------------------
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    st.error("Error loading model. Check logs.")
    st.stop()

# -------------------------------
# Load Encoders
# -------------------------------
try:
    with open(ENCODERS_PATH, "rb") as f:
        encoders_bundle = pickle.load(f)
    logger.info("Encoders loaded successfully.")
except Exception as e:
    logger.error(f"Error loading encoders: {e}")
    st.error("Error loading encoders. Check logs.")
    st.stop()

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="Car Price Prediction 🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("Car Price Predictor")
st.sidebar.markdown("""
This app predicts the **Selling Price** of a car.
- Enter car details
- Click **Predict**
- View results below
""")

# -------------------------------
# Main Page
# -------------------------------
st.title("🚗 Car Price Prediction")
st.markdown(
    "Enter the details of your car below to see the estimated selling price."
)

# -------------------------------
# Input Form with validation ranges
# -------------------------------
with st.form(key="car_form"):
    st.subheader("Car Features")

    # Categorical options from trained encoders
    car_name_options = encoders_bundle['encoders']['Car_Name'].classes_
    car_name = st.selectbox("Car Name", options=car_name_options)

    fuel_type_options = encoders_bundle['encoders']['Fuel_Type'].classes_
    fuel_type = st.selectbox("Fuel Type", options=fuel_type_options)

    selling_type_options = encoders_bundle['encoders']['Selling_type'].classes_
    selling_type = st.selectbox("Selling Type", options=selling_type_options)

    transmission_options = encoders_bundle['encoders']['Transmission'].classes_
    transmission = st.selectbox("Transmission", options=transmission_options)

    # Numeric fields with min/max limits based on training data
    present_price = st.number_input("Present Price (in Lakhs)", min_value=0.0, max_value=50.0, step=0.01, format="%.2f")
    driven_kms = st.number_input("Driven KMs", min_value=0, max_value=500000, step=100)
    owner = st.number_input("Owner (0/1/2/3)", min_value=0, max_value=3, step=1)
    car_year = st.number_input("Year of Purchase", min_value=2000, max_value=2025, step=1)

    submit_button = st.form_submit_button("Predict")

# -------------------------------
# Prediction with validation
# -------------------------------
if submit_button:
    errors = []

    # Additional runtime checks (extra safety)
    if present_price < 0 or present_price > 50:
        errors.append("Present Price is out of valid range (0-50 Lakhs).")
    if driven_kms < 0 or driven_kms > 500000:
        errors.append("Driven KMs is out of valid range (0-500,000).")
    if owner not in [0, 1, 2, 3]:
        errors.append("Owner must be 0, 1, 2, or 3.")
    if car_year < 2000 or car_year > 2025:
        errors.append("Year of Purchase must be between 2000 and 2025.")

    if errors:
        for err in errors:
            st.error(f"⚠️ {err}")
        st.stop()  # Prevent prediction until corrected

    try:
        # Prepare input dict
        input_dict = {
            "Car_Name": car_name,
            "Present_Price": present_price,
            "Driven_kms": driven_kms,
            "Fuel_Type": fuel_type,
            "Selling_type": selling_type,
            "Transmission": transmission,
            "Owner": owner,
            "Car_Age": 2025 - car_year
        }

        # Preprocess input safely
        processed_input = preprocess_input(input_dict, encoders_bundle)

        # Predict price
        pred = predict_price(model, processed_input)
        predicted_price = max(round(float(pred[0]), 2), 0)  # prevent negative price

        st.success(f"💰 Predicted Selling Price: ₹ {predicted_price} Lakhs")
        logger.info(f"Prediction successful: {predicted_price} Lakhs")

        # Show entered details
        st.markdown("📋 **Entered Car Details**")
        st.dataframe(pd.DataFrame([{
            "Car_Name": car_name,
            "Present_Price (Lakhs)": present_price,
            "Driven_KMs": driven_kms,
            "Fuel_Type": fuel_type,
            "Selling_Type": selling_type,
            "Transmission": transmission,
            "Owner": owner,
            "Year_of_Purchase": car_year
        }]))

    except Exception as e:
        st.error("⚠️ Error during prediction. Check logs.")
        logger.error(f"Prediction error: {e}")
