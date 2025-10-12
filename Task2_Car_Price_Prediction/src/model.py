import logging
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Initialize logger
logger = logging.getLogger(__name__)

def train_model(df, target_col="Selling_Price"):
    """
    Train a Linear Regression model on the provided dataframe.

    Parameters:
        df (pd.DataFrame): Feature-engineered DataFrame
        target_col (str): Name of the target column

    Returns:
        model (sklearn.linear_model.LinearRegression): Trained model
    """
    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)

        logger.info(f"Model trained. R2: {r2:.4f}, MSE: {mse:.4f}")
        print(f" Model trained. R2: {r2:.4f}, MSE: {mse:.4f}")

        return model

    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        raise

def save_model(model, path):
    """
    Save the trained model to the specified path.

    Parameters:
        model: Trained sklearn model
        path (str): File path to save the model
    """
    try:
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved at {path}")
        print(f" Model saved at {path}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_model(path):
    """
    Load a trained model from the specified path.

    Parameters:
        path (str): File path to load the model from

    Returns:
        model: Loaded sklearn model
    """
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        print(f" Model loaded from {path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise