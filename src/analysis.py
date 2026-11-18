"""Helpers for the dashboard charts."""
import re
from collections import Counter

import pandas as pd

from .preprocess import STOPWORDS

# Words in almost every Alexa review that say little about sentiment
DOMAIN_STOPWORDS = {'alexa', 'echo', 'dot', 'amazon', 'device', 'one', 'get', 'use'}


def top_words(texts, n=15):
    """Most frequent meaningful words, left unstemmed so they stay readable."""
    counter = Counter()
    for text in texts:
        for word in re.sub('[^a-zA-Z]', ' ', str(text)).lower().split():
            if len(word) > 2 and word not in STOPWORDS and word not in DOMAIN_STOPWORDS:
                counter[word] += 1
    return pd.DataFrame(counter.most_common(n), columns=['word', 'count'])
