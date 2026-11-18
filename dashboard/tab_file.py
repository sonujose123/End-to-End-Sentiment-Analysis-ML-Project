"""Tab 2: score a whole file of reviews."""
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.common import COLORS, load_sample, score


def render():
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

    if df is None:
        return

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
