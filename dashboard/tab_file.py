"""Tab 2: score a whole file of reviews and explore the results."""
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.common import COLORS, load_sample, score
from src.analysis import top_words

NONE = '(none)'


def _index(options, name):
    return options.index(name) if name in options else 0


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

    # Let the user say which column is which (we guess from common names)
    columns = list(df.columns)
    options = [NONE] + columns
    c1, c2, c3, c4 = st.columns(4)
    text_col = c1.selectbox('Review text', columns, index=_index(columns, 'verified_reviews'))
    group_col = c2.selectbox('Product / group', options, index=_index(options, 'variation'))
    date_col = c3.selectbox('Date', options, index=_index(options, 'date'))
    rating_col = c4.selectbox('Star rating', options, index=_index(options, 'rating'))

    # Score every review and attach the results to the original columns
    data = df.dropna(subset=[text_col]).reset_index(drop=True)
    results = score(tuple(data[text_col].astype(str)))
    out = pd.concat([data, results], axis=1)

    # Summary numbers and sentiment split
    share_pos = (out['sentiment'] == 'positive').mean()
    k1, k2, k3 = st.columns(3)
    k1.metric('Reviews analyzed', f'{len(out):,}')
    k2.metric('Positive', f'{share_pos:.1%}')
    k3.metric('Negative', f'{1 - share_pos:.1%}')

    left, right = st.columns(2)
    counts = out['sentiment'].value_counts().rename_axis('sentiment').reset_index(name='count')
    fig = px.pie(counts, names='sentiment', values='count', hole=0.55,
                 color='sentiment', color_discrete_map=COLORS, title='Sentiment split')
    left.plotly_chart(fig)

    if rating_col != NONE:
        stars = out.assign(stars=pd.to_numeric(out[rating_col], errors='coerce')).dropna(subset=['stars'])
        by_star = stars.groupby(['stars', 'sentiment']).size().reset_index(name='count')
        fig = px.bar(by_star, x='stars', y='count', color='sentiment',
                     color_discrete_map=COLORS, title='Predicted sentiment by star rating')
        right.plotly_chart(fig)
    else:
        fig = px.histogram(out, x='probability_positive', nbins=20,
                           title='How sure is the model? (probability of positive)')
        right.plotly_chart(fig)

    # Share of negative reviews per product
    if group_col != NONE:
        out[group_col] = out[group_col].astype(str).str.strip()
        g = out.groupby([group_col, 'sentiment']).size().unstack(fill_value=0)
        for label in ('positive', 'negative'):
            if label not in g:
                g[label] = 0
        g['% negative'] = g['negative'] / (g['positive'] + g['negative']) * 100
        g = g.sort_values('% negative').reset_index()
        fig = px.bar(g, x='% negative', y=group_col, orientation='h',
                     hover_data=['positive', 'negative'],
                     title=f'Share of negative reviews by {group_col}',
                     color_discrete_sequence=[COLORS['negative']])
        st.plotly_chart(fig)

    # Reviews over time
    if date_col != NONE:
        dates = pd.to_datetime(out[date_col], errors='coerce', format='mixed', dayfirst=True)
        valid = dates.notna()
        if valid.sum() > 1:
            span_days = (dates[valid].max() - dates[valid].min()).days
            freq = 'M' if span_days > 120 else 'W'
            period = dates[valid].dt.to_period(freq).dt.start_time.rename('period')
            over_time = (out[valid].assign(period=period)
                         .groupby(['period', 'sentiment']).size().reset_index(name='count'))
            label = 'monthly' if freq == 'M' else 'weekly'
            fig = px.bar(over_time, x='period', y='count', color='sentiment',
                         color_discrete_map=COLORS, title=f'Reviews over time ({label})')
            st.plotly_chart(fig)

    # Most common words in each group
    w1, w2 = st.columns(2)
    for column, label in ((w1, 'negative'), (w2, 'positive')):
        words = top_words(out.loc[out['sentiment'] == label, text_col])
        if not words.empty:
            fig = px.bar(words.iloc[::-1], x='count', y='word', orientation='h',
                         title=f'Top words in {label} reviews',
                         color_discrete_sequence=[COLORS[label]])
            column.plotly_chart(fig)

    # Browse, search and download
    st.subheader('Browse reviews')
    f1, f2 = st.columns([1, 2])
    pick = f1.multiselect('Sentiment', ['positive', 'negative'], default=['positive', 'negative'])
    query = f2.text_input('Search in review text')
    view = out[out['sentiment'].isin(pick)]
    if query:
        view = view[view[text_col].astype(str).str.contains(query, case=False, na=False, regex=False)]
    view = view.sort_values('probability_positive')
    st.caption(f'{len(view):,} reviews, most negative first')
    st.dataframe(view)
    st.download_button('Download results (CSV)', out.to_csv(index=False).encode('utf-8'),
                       'sentiment_results.csv', 'text/csv')
