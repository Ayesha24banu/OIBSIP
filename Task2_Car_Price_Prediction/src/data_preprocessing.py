import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

def load_and_clean_data(file_path: str, save_path: str = None):
    """
    Load CSV, clean missing values, and save processed file.
    """
    try:
        # Load data
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        print(f" Data loaded: {df.shape}")

        # Drop duplicates
        df = df.drop_duplicates()

        # Handle missing values
        df = df.dropna()

        logger.info(f"Data cleaned. Final shape: {df.shape}")
        print(f" Data cleaned: {df.shape}")

        # Save processed data
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            logger.info(f"Processed data saved at {save_path}")
            print(f" Processed data saved at {save_path}")

        return df

    except Exception as e:
        logger.error(f"Error in load_and_clean_data: {str(e)}")
        raise
