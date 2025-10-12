# src/predict.py
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def predict_price(model, df: pd.DataFrame):
    """
    Predict car prices using a trained model.

    Parameters:
        model : sklearn trained model
        df : pd.DataFrame of features (must match training features)

    Returns:
        predictions : numpy array of predicted prices
    """
    try:
        preds = model.predict(df)
        print(" Prediction completed successfully.")
        return preds
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        print(f" Prediction error: {str(e)}")
        raise

def preprocess_input(input_dict, encoders_bundle: dict):
    """
    Preprocess input dictionary using saved LabelEncoders.
    Handles unseen labels by assigning -1.

    Parameters:
        input_dict (dict): Single row of input features
        encoders_bundle (dict): Contains {"encoders", "categorical_cols"}

    Returns:
        df (pd.DataFrame): Preprocessed input ready for prediction
    """
    try:
        encoders = encoders_bundle.get("encoders", {})
        categorical_cols = encoders_bundle.get("categorical_cols", [])

        # Convert input to DataFrame if dict
        if isinstance(input_dict, dict):
            df = pd.DataFrame([input_dict])
        else:
            df = pd.DataFrame(input_dict)

        # Validate missing columns
        missing = [c for c in categorical_cols if c not in df.columns]
        if missing:
            msg = f"Missing categorical feature(s): {missing}"
            logger.error(msg)
            print(f" {msg}")
            raise ValueError(msg)

        # Apply LabelEncoders only to categorical columns
        for col in categorical_cols:
            if df[col].dtype == object or df[col].dtype == 'O':
                le = encoders[col]
                # Handle unseen labels by mapping them to -1
                df[col] = df[col].apply(lambda x: le.transform([x])[0] 
                                        if x in le.classes_ else -1)

        print(" Input preprocessing completed successfully.")
        return df

    except Exception as e:
        logger.error(f"Error in preprocess_input: {str(e)}")
        print(f" Error in preprocess_input: {str(e)}")
        raise
