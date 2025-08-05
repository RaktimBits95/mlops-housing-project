import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import mlflow
import mlflow.sklearn

# ✅ Connect to MLflow tracking server (make sure mlflow ui is running)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# ✅ Load training data
train_path = "data/prepared/train.csv"
df = pd.read_csv(train_path)

# ✅ Separate input and output
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# ✅ Make sure models/ directory exists
os.makedirs("models", exist_ok=True)

# ✅ List of models to train
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42)
}

# ✅ Train each model and log to MLflow
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Train
        model.fit(X, y)
        preds = model.predict(X)
        rmse = sqrt(mean_squared_error(y, preds))

        # Log parameters and metrics
        mlflow.log_param("model_name", name)
        if name == "DecisionTree":
            mlflow.log_param("max_depth", 5)
        mlflow.log_metric("rmse", rmse)

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")

        # Save model locally (for DVC to track)
        model_path = f"models/model_{name}.pkl"
        joblib.dump(model, model_path)

print("✅ Training complete. Models saved and logged to MLflow.")
