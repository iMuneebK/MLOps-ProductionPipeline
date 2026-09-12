from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import uvicorn
from src.prediction_pipeline import PredictionPipeline

app = FastAPI(title="MLOps Prediction Service")
pipeline = PredictionPipeline()

class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float

@app.get("/")
def home():
    return {"message": "Welcome to the MLOps Pipeline Prediction API"}

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    prediction = pipeline.predict(df)
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
