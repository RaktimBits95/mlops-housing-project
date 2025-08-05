import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load test data
df = pd.read_csv("data/prepared/test.csv")
X_test = df.drop("MedHouseVal", axis=1)
y_test = df["MedHouseVal"]

# Evaluate both models
model_files = {
    "LinearRegression": "models/model_LinearRegression.pkl",
    "DecisionTree": "models/model_DecisionTree.pkl"
}

with open("metrics.txt", "w") as f:
    for name, path in model_files.items():
        model = joblib.load(path)
        preds = model.predict(X_test)
        rmse = sqrt(mean_squared_error(y_test, preds))
        f.write(f"{name} RMSE: {rmse:.4f}\n")
