"""Streamlit dashboard for review sentiment.

Run from the project root:
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from dashboard import tab_file, tab_model, tab_single

st.set_page_config(page_title='Review Sentiment', page_icon='🔊', layout='wide')
st.title('Review Sentiment Dashboard')

t1, t2, t3 = st.tabs(['Try a review', 'Analyze a file', 'Model performance'])
with t1:
    tab_single.render()
with t2:
    tab_file.render()
with t3:
    tab_model.render()
