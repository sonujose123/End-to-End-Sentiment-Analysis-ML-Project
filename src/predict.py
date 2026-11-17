"""Load the saved model files and score new reviews."""
import pickle

from .config import LABELS, MODELS_DIR
from .preprocess import clean_text


def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class SentimentPredictor:
    """Loads the vectorizer, scaler and model once, then scores text."""

    def __init__(self, models_dir=MODELS_DIR):
        try:
            self.vectorizer = _load(models_dir / 'count_vectorizer.pkl')
            self.scaler = _load(models_dir / 'scaler.pkl')
            self.model = _load(models_dir / 'model.pkl')
        except FileNotFoundError as e:
            raise FileNotFoundError(
                'Model files not found. Put the three .pkl files in the models folder.'
            ) from e

    def predict(self, texts, batch_size=1000):
        """Score one review (a string) or many (a list). Returns a list of dicts."""
        if isinstance(texts, str):
            texts = [texts]
        texts = [str(t) for t in texts]

        results = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            cleaned = [clean_text(t) for t in chunk]
            X = self.scaler.transform(self.vectorizer.transform(cleaned).toarray())
            probabilities = self.model.predict_proba(X)
            for text, p in zip(chunk, probabilities):
                label = int(p.argmax())
                results.append({
                    'text': text,
                    'sentiment': LABELS[label],
                    'confidence': round(float(p[label]), 4),
                    'probability_positive': round(float(p[1]), 4),
                })
        return results


if __name__ == '__main__':
    import sys

    review = ' '.join(sys.argv[1:]) or 'I love my Echo, it works great!'
    print(SentimentPredictor().predict(review)[0])
