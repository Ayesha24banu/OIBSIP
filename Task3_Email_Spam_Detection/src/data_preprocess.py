import pandas as pd
import re
import string
import os
from sklearn.model_selection import train_test_split

# Text Cleaning Function
def clean_text(text):
    """
    Clean text: lowercase, remove punctuation, numbers, extra spaces
    """
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Preprocess dataset
def preprocess_data(df, logger):
    """
    Apply text cleaning and encode labels
    """
    df['text'] = df['text'].apply(clean_text)
    df['label_enc'] = df['label'].map({'ham': 0, 'spam': 1})
    if logger:
        logger.info(f"Dataset preprocessed: {df.shape[0]} rows")
    print(f"Dataset preprocessed: {df.shape[0]} rows")
    return df

# Save processed dataset
def save_processed_data(df, path, logger):
    """
    Save processed dataframe to disk
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    if logger:
        logger.info(f"Processed dataset saved to {path}")
    print(f"Processed dataset saved to {path}")

# Split data into train/test
def split_data(df,  logger, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    """
    X = df['text']
    y = df['label_enc']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    if logger:
        logger.info(f"Data split into train ({X_train.shape[0]}) and test ({X_test.shape[0]})")
    return X_train, X_test, y_train, y_test