import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def train_model(data_path: str):
    mlflow.set_tracking_uri("file:./experiments")
    mlflow.set_experiment("california-housing")

    df = pd.read_csv(data_path)

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        # Log params
        mlflow.log_param("n_estimators", 100)

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Log model artifact
        mlflow.sklearn.log_model(model, "model")

    return {
        "rmse": rmse,
        "r2": r2
    }
