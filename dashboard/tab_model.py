"""Tab 3: how well does the model work?"""
import json

import plotly.express as px
import streamlit as st

from src.config import MODELS_DIR


def render():
    st.subheader('How well does the model work?')
    path = MODELS_DIR / 'metrics.json'
    if not path.exists():
        st.warning('No results found yet. Run this in the Terminal: python -m src.evaluate')
        return

    m = json.loads(path.read_text())
    cm = m['confusion_matrix']
    total = sum(sum(row) for row in cm)
    baseline = sum(cm[1]) / total  # accuracy of always answering 'positive'

    a, b, c, d = st.columns(4)
    a.metric('Accuracy', f'{m["accuracy"]:.1%}')
    b.metric('Always-positive baseline', f'{baseline:.1%}')
    c.metric('Negative recall', f'{m["negative"]["recall"]:.1%}')
    d.metric('Negative precision', f'{m["negative"]["precision"]:.1%}')

    fig = px.imshow(
        cm,
        x=['Predicted negative', 'Predicted positive'],
        y=['Actual negative', 'Actual positive'],
        text_auto=True,
        color_continuous_scale='Blues',
        title=f'Confusion matrix ({total} held-out test reviews)',
    )
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig)

    st.info(
        'About 92% of reviews are positive, so a model that always says positive is already '
        f'{baseline:.0%} accurate. Look at negative recall (how many unhappy reviews it finds) '
        'and negative precision (how often a negative flag is right) instead of accuracy alone.'
    )
