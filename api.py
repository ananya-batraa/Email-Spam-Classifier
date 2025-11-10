import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

model_path = "./spam_classifier.pkl"
vectorizer_path = "./tfidf_vectorizer.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

with open(model_path, "rb") as f:
    model = pickle.load(f)
    if not hasattr(model, "predict"):
        raise ValueError("Loaded model does not have a predict method.")
    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model does not have a predict_proba method.")
    
if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)
    if not hasattr(vectorizer, "transform"):
        raise ValueError("Loaded vectorizer does not have a transform method.")


app = FastAPI()
origins = [ "http://localhost:5500", "http://127.0.0.1:5500" ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str

class Prediction(BaseModel):
    label: str
    probability: float

class PredictionResponse(BaseModel):
    message: str
    predictions: List[Prediction]

@app.get("/")
def root():
    return {"message": "Welcome to the Spam Classifier API!"}

@app.post("/predict", response_model=PredictionResponse)
def predict(message: Message):
    if not isinstance(message, Message):
        raise HTTPException(status_code=400, detail="Invalid input format. Expected a Message object.")
    
    text = message.text
    if not isinstance(text, str):
        raise HTTPException(status_code=400, detail="Text must be a string.")
    
    text_vectorized = vectorizer.transform([text])
    
    prediction = model.predict(text_vectorized)
    prediction_proba = model.predict_proba(text_vectorized)
    
    label = "spam" if prediction[0] == 1 else "ham"
    probability = np.max(prediction_proba[0])
    
    return PredictionResponse(
        message="Prediction successful",
        predictions=[Prediction(label=label, probability=probability)]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)