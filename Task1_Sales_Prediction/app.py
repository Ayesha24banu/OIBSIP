# =========================================
# Advertising Sales Prediction App
# =========================================

import streamlit as st
import os
import matplotlib.pyplot as plt
from src.deploy import load_model, predict_sales

# Page Config
st.set_page_config(
    page_title="Advertising Sales Prediction",
    page_icon="📈",
    layout="centered",
)

# App Header
st.title("📊 Advertising Sales Prediction")
st.markdown(
    """
    Enter the advertising spend details below to predict **Sales**.
    The model uses historical data with engineered features for accurate prediction.
    """
)

# Input Section - Main Page
st.subheader("Enter Advertising Spend")
tv = st.number_input("TV Advertising Spend", min_value=0.0, value=100.0, step=1.0)
radio = st.number_input("Radio Advertising Spend", min_value=0.0, value=30.0, step=1.0)
newspaper = st.number_input("Newspaper Advertising Spend", min_value=0.0, value=20.0, step=1.0)

# Derived Features
total_ads = tv + radio + newspaper
tv_radio = tv * radio
tv_newspaper = tv * newspaper
radio_newspaper = radio * newspaper

input_features = {
    "TV": tv,
    "Radio": radio,
    "Newspaper": newspaper,
    "Total_Ads": total_ads,
    "TV_Radio": tv_radio,
    "TV_Newspaper": tv_newspaper,
    "Radio_Newspaper": radio_newspaper,
}

# Load Model
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
model, scaler = load_model(model_path)

# Prediction Button
if st.button("Predict Sales"):
    predicted_sales = predict_sales(model, scaler, input_features)
    
    st.success(f"📈 Predicted Sales: {predicted_sales:.2f}")
    
    # Display entered + derived features
    st.subheader("Entered & Derived Features")
    st.json(input_features)

    # Visualization - Spend vs Predicted Sales
    st.subheader("Visualization")
    fig, ax = plt.subplots(figsize=(7,4))
    
    categories = ["TV", "Radio", "Newspaper", "Predicted Sales"]
    values = [tv, radio, newspaper, predicted_sales]
    colors = ["skyblue", "lightgreen", "salmon", "gold"]

    bars = ax.bar(categories, values, color=colors)
    ax.set_ylabel("Amount")
    ax.set_title("Advertising Spend vs Predicted Sales")
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height:.2f}', ha='center', fontsize=10)
    
    st.pyplot(fig)

# Footer / Credits
st.markdown(
    """
    ---
    Developed by: **Ayesha Banu** | Data Science Project
    """
)
