"""Measure the saved model on the held-out test set and save the results.

Run from the project root:
    python -m src.evaluate
"""
import json

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, LABELS, MODELS_DIR
from .predict import SentimentPredictor
from .preprocess import clean_text


def main():
    df = pd.read_csv(DATA_PATH, delimiter='\t', quoting=3)
    df = df.dropna(subset=['verified_reviews']).reset_index(drop=True)

    predictor = SentimentPredictor()
    X = predictor.vectorizer.transform(df['verified_reviews'].apply(clean_text)).toarray()
    y = df['feedback'].values

    # Same split as in the notebook: same seed, same proportions
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=15, stratify=y)

    y_pred = predictor.model.predict(predictor.scaler.transform(X_test))
    report = classification_report(
        y_test, y_pred, target_names=[LABELS[0], LABELS[1]], output_dict=True)

    metrics = {
        'test_size': int(len(y_test)),
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'negative': report['negative'],
        'positive': report['positive'],
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }
    with open(MODELS_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print('Test reviews:', metrics['test_size'])
    print('Accuracy:', round(metrics['accuracy'], 3))
    print('Negative recall:', round(metrics['negative']['recall'], 3))
    print('Negative precision:', round(metrics['negative']['precision'], 3))
    print('Negative F1:', round(metrics['negative']['f1-score'], 3))
    print('Saved', MODELS_DIR / 'metrics.json')


if __name__ == '__main__':
    main()
