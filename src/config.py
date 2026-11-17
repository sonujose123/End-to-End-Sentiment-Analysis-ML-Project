"""Shared paths and settings for the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / 'data' / 'amazon_alexa.tsv'
MODELS_DIR = ROOT / 'models'

LABELS = {0: 'negative', 1: 'positive'}
