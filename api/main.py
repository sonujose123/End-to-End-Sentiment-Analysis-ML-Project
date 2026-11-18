"""REST API for the sentiment model."""
from functools import lru_cache
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import SentimentPredictor

app = FastAPI(
    title='Alexa Review Sentiment API',
    version='1.0.0',
    description='Classify product reviews as positive or negative.',
)


class ReviewIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class BatchIn(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=500)


class Prediction(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probability_positive: float


@lru_cache(maxsize=1)
def get_predictor():
    return SentimentPredictor()


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict', response_model=Prediction)
def predict(review: ReviewIn):
    return get_predictor().predict(review.text)[0]


@app.post('/predict/batch', response_model=List[Prediction])
def predict_batch(batch: BatchIn):
    return get_predictor().predict(batch.texts)
