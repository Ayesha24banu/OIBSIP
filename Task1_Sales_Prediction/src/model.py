# Model
import os
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)

# Train & Compare Models
def train_and_compare_models(df, target_col="Sales", test_size=0.2, random_state=42, model_save_path=None):
    """
    Train multiple regression models, compare performance, and save the best model.
    Models:
    - Linear Regression
    - Random Forest Regressor
    - Gradient Boosting Regressor
    """
    try:
        # Split data
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Define models
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(random_state=random_state),
            "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
        }

        results = {}
        best_model = None
        best_score = -float("inf")

        # Train & evaluate
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)

            results[name] = {"RMSE": rmse, "R2": r2}

            logger.info(f"{name} -> RMSE: {rmse:.3f}, R2: {r2:.3f}")

            if r2 > best_score:
                best_score = r2
                best_model = (name, model)

        # Save best model + scaler
        if model_save_path:
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            joblib.dump({"model": best_model[1], "scaler": scaler}, model_save_path)
            logger.info(f" Best model '{best_model[0]}' saved at {model_save_path}")

        return results, best_model[0]

    except Exception as e:
        logger.error(f"Error in training models: {e}")
        raise
