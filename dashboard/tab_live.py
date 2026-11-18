"""Tab 3: a simulated live feed of incoming reviews."""
import time
from collections import Counter

import pandas as pd
import streamlit as st

from dashboard.common import COLORS, load_predictor
from src.sources import SampleSource


def render():
    st.subheader('Live feed (simulated)')
    st.write(
        'Replays reviews from the sample dataset one at a time and scores each as it '
        'arrives, the way a real-time review stream would. To use a real source, add a '
        'class to `src/sources.py` that yields reviews.'
    )
    c1, c2 = st.columns(2)
    n = c1.slider('Reviews to stream', 20, 300, 80, step=10)
    delay = c2.slider('Seconds between reviews', 0.0, 1.0, 0.15, step=0.05)

    if not st.button('Start live feed', type='primary'):
        return

    predictor = load_predictor()
    source = SampleSource(seed=int(time.time()))
    kpis = st.empty()
    chart = st.empty()
    table = st.empty()

    seen = Counter()
    history = []
    feed = []
    for i, review in enumerate(source.iter_reviews(n), start=1):
        result = predictor.predict(review['text'])[0]
        seen[result['sentiment']] += 1
        history.append({'reviews so far': i, '% negative': seen['negative'] / i * 100})
        feed.insert(0, {
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'stars': review['rating'],
            'review': review['text'][:140],
        })

        with kpis.container():
            a, b, c = st.columns(3)
            a.metric('Reviews so far', i)
            b.metric('Positive', seen['positive'])
            c.metric('Negative', seen['negative'])
        chart.line_chart(pd.DataFrame(history).set_index('reviews so far'),
                         color=COLORS['negative'])
        table.dataframe(pd.DataFrame(feed[:10]))
        time.sleep(delay)

    st.success('Feed finished.')
