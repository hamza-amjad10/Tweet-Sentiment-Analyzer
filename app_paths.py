from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent
MODEL_DIRECTORY = REPOSITORY_ROOT / "model"
MODEL_PATH = MODEL_DIRECTORY / "model.pkl"
VECTORIZER_PATH = MODEL_DIRECTORY / "tfidf vectorizer.pkl"
DATASET_PATH = REPOSITORY_ROOT / "test.csv"
