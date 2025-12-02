from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="News Text Classifier API")

# Load trained pipeline
model = joblib.load("news_text_classifier.joblib")
target_names = ["rec.sport.baseball", "sci.space"]  # same as training

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_category(input: TextInput):
    pred = model.predict([input.text])[0]
    return {
        "predicted_label": int(pred),
        "predicted_category": target_names[pred],
    }
