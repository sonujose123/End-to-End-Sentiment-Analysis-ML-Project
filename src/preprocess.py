"""Text cleaning shared by training and prediction."""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Negations flip a review's meaning ('not good'), so we keep them
# even though NLTK lists them as stopwords.
NEGATIONS = {
    'no', 'nor', 'not', 'don', 'didn', 'doesn', 'isn', 'wasn', 'weren',
    'won', 'wouldn', 'couldn', 'shouldn', 'hasn', 'haven', 'hadn',
    'aren', 'ain', 'mustn', 'needn', 'shan', 'mightn',
}


def _load_stopwords():
    try:
        words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        words = set(stopwords.words('english'))
    return words - NEGATIONS


STOPWORDS = _load_stopwords()
_stemmer = PorterStemmer()


def clean_text(text):
    """Lowercase, strip non-letters, drop stopwords, and stem."""
    tokens = re.sub('[^a-zA-Z]', ' ', str(text)).lower().split()
    return ' '.join(_stemmer.stem(t) for t in tokens if t not in STOPWORDS)
