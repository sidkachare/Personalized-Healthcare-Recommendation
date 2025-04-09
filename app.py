from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("model.pkl")

app = FastAPI(title="Healthcare Prediction API")
class HealthInput(BaseModel):
    features: list

@app.get("/")
def root():
    return {"message": "Healthcare Prediction API is running."}

@app.post("/predict/")
def predict(data: HealthInput):
    input_array = np.array(data.features).reshape(1, -1)

    prediction = model.predict(input_array)[0]
    return {"prediction": int(prediction)}
