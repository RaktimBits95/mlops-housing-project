from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os
import uvicorn
import logging
from datetime import datetime

# Setup logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "app.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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
    with open("app/index.html", "r" , encoding='utf-8') as f:
        return f.read()

@app.post("/predict")
def predict(data: HouseData):
    try:
        input_data = np.array([[ 
            data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms, 
            data.Population, data.AveOccup, data.Latitude, data.Longitude
        ]])
        prediction = model.predict(input_data)[0]

        # 🔧 Log input and output
        logging.info(f"Prediction input: {data.dict()}")
        logging.info(f"Predicted price: {prediction}")


        return {"predicted_price": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/metrics")
def metrics():
    return {
        "model_loaded": True,
        "model_path": MODEL_PATH,
        "log_file": os.path.abspath(os.path.join(LOG_DIR, "app.log"))
    }
