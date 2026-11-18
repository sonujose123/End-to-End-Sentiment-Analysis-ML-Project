"""Streamlit dashboard for review sentiment."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import DATA_PATH
from src.predict import SentimentPredictor

st.set_page_config(page_title='Review Sentiment', page_icon='🔊', layout='wide')

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


st.title('Review Sentiment Dashboard')
tab_single, tab_file = st.tabs(['Try a review', 'Analyze a file'])

with tab_single:
    st.subheader('Score a single review')
    text = st.text_area('Review text', height=120, placeholder='Type or paste a review...')
    if text.strip():
        result = load_predictor().predict(text)[0]
        p_pos = result['probability_positive']
        st.subheader(result['sentiment'].title())
        st.progress(p_pos, text=f'Probability positive: {p_pos:.1%}')
        st.caption(f'Model confidence: {result["confidence"]:.1%}')

with tab_file:
    st.subheader('Analyze many reviews at once')
    uploaded = st.file_uploader('Upload a CSV or TSV of reviews', type=['csv', 'tsv'])
    use_sample = st.checkbox('Use the sample Alexa dataset', value=uploaded is None)

    df = None
    if uploaded is not None:
        if uploaded.name.lower().endswith('.tsv'):
            df = pd.read_csv(uploaded, sep='\t', quoting=3)
        else:
            df = pd.read_csv(uploaded)
    elif use_sample:
        df = load_sample()

    if df is not None:
        columns = list(df.columns)
        default = columns.index('verified_reviews') if 'verified_reviews' in columns else 0
        column = st.selectbox('Which column holds the review text?', columns, index=default)

        reviews = df[column].dropna().astype(str).reset_index(drop=True)
        out = score(tuple(reviews)).copy()
        out.insert(0, 'review', reviews)

        share_pos = (out['sentiment'] == 'positive').mean()
        c1, c2, c3 = st.columns(3)
        c1.metric('Reviews analyzed', f'{len(out):,}')
        c2.metric('Positive', f'{share_pos:.1%}')
        c3.metric('Negative', f'{1 - share_pos:.1%}')

        counts = out['sentiment'].value_counts().rename_axis('sentiment').reset_index(name='count')
        fig = px.pie(counts, names='sentiment', values='count', hole=0.55,
                     color='sentiment', color_discrete_map=COLORS)
        st.plotly_chart(fig)
        st.dataframe(out)
