from pathlib import Path
from typing import cast

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from app_paths import DATASET_PATH, MODEL_PATH, VECTORIZER_PATH
from sentiment import clean_text


def train_model() -> tuple[TfidfVectorizer, RandomForestClassifier]:
    data = pd.read_csv(DATASET_PATH, encoding="latin1")[["text", "sentiment"]]
    data.dropna(inplace=True)
    text = cast(pd.Series, data["text"]).apply(clean_text)
    labels = cast(pd.Series, data["sentiment"]).map(
        {"positive": 1, "negative": 2, "neutral": 0}
    )

    training_text, test_text, training_labels, test_labels = train_test_split(
        text,
        labels,
        test_size=0.2,
        random_state=42,
    )

    vectorizer = TfidfVectorizer()
    vectorizer.fit(training_text)
    training_vectors = vectorizer.transform(training_text)
    test_vectors = vectorizer.transform(test_text)

    model = RandomForestClassifier(random_state=42)
    model.fit(training_vectors, training_labels)
    predictions = model.predict(test_vectors)

    print("Confusion matrix:", confusion_matrix(test_labels, predictions))
    print("Classification report:", classification_report(test_labels, predictions))
    print("Accuracy score:", accuracy_score(test_labels, predictions))
    return vectorizer, model


def save_artifacts(
    vectorizer: TfidfVectorizer,
    model: RandomForestClassifier,
    vectorizer_path: Path = VECTORIZER_PATH,
    model_path: Path = MODEL_PATH,
) -> None:
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model, model_path)


def main() -> None:
    vectorizer, model = train_model()
    save_artifacts(vectorizer, model)


if __name__ == "__main__":
    main()
