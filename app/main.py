from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Define FastAPI app
app = FastAPI(title="Housing Price Predictor")

# Load model
MODEL_PATH = "models/model_DecisionTree.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# Define input format
class HouseData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/", response_class=HTMLResponse)
def root():
    with open("app/index.html", "r") as f:
        return f.read()

@app.post("/predict")
def predict(data: HouseData):
    try:
        input_data = np.array([[ 
            data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms, 
            data.Population, data.AveOccup, data.Latitude, data.Longitude
        ]])
        prediction = model.predict(input_data)[0]
        return {"predicted_price": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
