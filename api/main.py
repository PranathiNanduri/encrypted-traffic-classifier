from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from src.inference.predictor import Predictor

app = FastAPI(title="Encrypted Traffic Classifier API")
predictor = Predictor()

class PredictRequest(BaseModel):
    x_len: List[float] = Field(..., description="Packet length sequence of size 50")
    x_iat: List[float] = Field(..., description="Inter-arrival time sequence of size 50")

@app.get("/")
def home():
    return {"status": "API is running"}

@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.x_len) != 50 or len(req.x_iat) != 50:
        return {"error": "x_len and x_iat must each be length 50"}

    return predictor.predict(req.x_len, req.x_iat)
