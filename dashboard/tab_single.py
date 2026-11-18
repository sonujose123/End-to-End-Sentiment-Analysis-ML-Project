"""Tab 1: score a single review."""
import streamlit as st

from dashboard.common import load_predictor


def render():
    st.subheader('Score a single review')
    text = st.text_area('Review text', height=120, placeholder='Type or paste a review...')
    if text.strip():
        result = load_predictor().predict(text)[0]
        p_pos = result['probability_positive']
        st.subheader(result['sentiment'].title())
        st.progress(p_pos, text=f'Probability positive: {p_pos:.1%}')
        st.caption(f'Model confidence: {result["confidence"]:.1%}')
