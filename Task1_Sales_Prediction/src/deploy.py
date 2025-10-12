# Deploy
import joblib
import numpy as np

def load_model(model_path):
    """
    Load the trained model + scaler from file.
    """
    data = joblib.load(model_path)
    return data["model"], data["scaler"]

def predict_sales(model, scaler, input_data):
    """
    Predict sales for new input data.
    input_data: dict with keys = feature names
    """
    features = np.array([list(input_data.values())]).astype(float)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return prediction[0]
