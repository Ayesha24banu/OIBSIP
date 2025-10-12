# Data_processing
import os
import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Get logger
logger = logging.getLogger(__name__)

# Load Data
def load_data(file_path):
    """
    Load dataset from CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f" Data loaded successfully from {file_path} | Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f" File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f" Error loading data: {e}")
        raise

# Preprocess Data
def preprocess_data(df, save_path=None):
    """
    Preprocess dataset:
    - Drop duplicates
    - Drop missing values
    - Drop unwanted index column
    - Save processed dataset if save_path provided
    """
    try:
        original_shape = df.shape

        # Drop duplicates and NA
        df = df.drop_duplicates().dropna()

        # Remove unwanted index column if present
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        logger.info(f" Preprocessing complete. Shape {original_shape} -> {df.shape}")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            logger.info(f" Processed data saved at: {save_path}")

        return df
    except Exception as e:
        logger.error(f" Error in preprocessing: {e}")
        raise