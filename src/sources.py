"""Where reviews come from.

Anything that yields dicts like {'text': ..., 'date': ..., 'variation': ..., 'rating': ...}
can feed the live view. Only 'text' is required.

To use a real source, write a new class with an iter_reviews() method that
yields reviews, for example from an official reviews API or a queue.

Note: Amazon and Flipkart restrict automated scraping in their terms of service.
Prefer official APIs or datasets you have permission to use.
"""
import random

import pandas as pd

from .config import DATA_PATH


class SampleSource:
    """Replays reviews from the bundled Alexa dataset in random order,
    to simulate a live stream of incoming reviews."""

    def __init__(self, path=DATA_PATH, seed=None):
        df = pd.read_csv(path, delimiter='\t', quoting=3)
        self.df = df.dropna(subset=['verified_reviews']).reset_index(drop=True)
        self.rng = random.Random(seed)

    def iter_reviews(self, n=100):
        order = list(range(len(self.df)))
        self.rng.shuffle(order)
        for i in order[:n]:
            row = self.df.iloc[i]
            yield {
                'text': row['verified_reviews'],
                'date': row['date'],
                'variation': str(row['variation']).strip(),
                'rating': int(row['rating']),
            }
