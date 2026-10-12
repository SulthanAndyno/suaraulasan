import os
from functools import cache

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from suaraulasan.train import LABELS

app = FastAPI(title="SuaraUlasan")


@cache
def model():
    return joblib.load(os.environ.get("MODEL_PATH", "models/model.joblib"))


class Review(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(r: Review):
    proba = model().predict_proba([r.text])[0]
    i = int(proba.argmax())
    return {"label": LABELS[i], "confidence": round(float(proba[i]), 4)}
