# src/feature_engineering.py
import pandas as pd
import logging
import pickle
import os
from sklearn.preprocessing import LabelEncoder

# Initialize logger
logger = logging.getLogger(__name__)

def add_car_age(df: pd.DataFrame, current_year: int = 2025) -> pd.DataFrame:
    """
    Add a Car_Age column based on the Year column and drop the original Year column.
    """
    try:
        logger.info("Adding Car_Age feature")
        df['Car_Age'] = current_year - df['Year']
        df.drop(columns=['Year'], inplace=True)
        logger.info("Car_Age feature added successfully.")
        return df
    except Exception as e:
        logger.error(f"Error in add_car_age: {str(e)}")
        raise

def encode_features(df: pd.DataFrame, save_path: str) -> pd.DataFrame:
    """
    Encode categorical features using LabelEncoder and save encoders.
    
    Parameters:
        df (pd.DataFrame): DataFrame with categorical columns
        save_path (str): Path to save encoders bundle

    Returns:
        df (pd.DataFrame): Encoded DataFrame
    """
    try:
        logger.info("Encoding categorical features")
        df = df.copy()
        encoders = {}
        categorical_cols = list(df.select_dtypes(include=['object']).columns)

        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
            logger.info(f"{col} encoded successfully.")

        # Bundle encoders and categorical column names
        encoders_bundle = {"encoders": encoders, "categorical_cols": categorical_cols}

        # Save bundle if path provided
        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(encoders_bundle, f)
            logger.info(f"Encoders saved at {save_path}")
            print(f" Encoders saved at {save_path}")
            
        print(" Feature engineering (Label Encoding) completed successfully.")
        return df

    except Exception as e:
        logger.error(f"Error in encode_features: {str(e)}")
        print(f" Error in encode_features: {str(e)}")
        raise
