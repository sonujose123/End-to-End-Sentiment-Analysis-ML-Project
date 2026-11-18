"""Shared helpers for the dashboard tabs."""
import pandas as pd
import streamlit as st

from src.config import DATA_PATH
from src.predict import SentimentPredictor

COLORS = {'positive': '#2E9E6B', 'negative': '#D64545'}


@st.cache_resource
def load_predictor():
    return SentimentPredictor()


@st.cache_data
def load_sample():
    df = pd.read_csv(DATA_PATH, delimiter='\t', quoting=3)
    return df.dropna(subset=['verified_reviews']).reset_index(drop=True)


@st.cache_data
def score(texts):
    results = load_predictor().predict(list(texts))
    return pd.DataFrame(results)[['sentiment', 'confidence', 'probability_positive']]
