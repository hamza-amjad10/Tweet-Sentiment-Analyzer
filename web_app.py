import io
import re
from collections.abc import Callable, Iterable
from typing import Any, BinaryIO

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from app_paths import MODEL_PATH, VECTORIZER_PATH
from sentiment import clean_text

MAX_CSV_BYTES = 10 * 1024 * 1024
MAX_CSV_ROWS = 500
PREFERRED_TEXT_COLUMN_NAMES = (
    "tweet text",
    "text",
    "tweet",
    "content",
    "message",
)
SENTIMENT_LABELS = {0: "neutral", 1: "positive", 2: "negative"}
SPREADSHEET_FORMULA_PATTERN = re.compile(r"^[\t\r\n=+\-@]|^ +[=+\-@]")


class CsvValidationError(ValueError):
    """Raised when an uploaded CSV cannot be processed safely."""


@st.cache_resource
def load_artifacts() -> tuple[Any, Any]:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def _normalized_column_name(name: object) -> str:
    normalized = re.sub(r"[_-]+", " ", str(name).strip().lower())
    return re.sub(r"\s+", " ", normalized)


def find_text_columns(data: pd.DataFrame) -> list[str]:
    normalized_columns: dict[str, list[str]] = {}
    for column in data.columns:
        normalized_columns.setdefault(_normalized_column_name(column), []).append(
            str(column)
        )

    preferred = [
        column
        for name in PREFERRED_TEXT_COLUMN_NAMES
        for column in normalized_columns.get(name, [])
    ]
    if preferred:
        return preferred

    return [
        str(column)
        for column in data.columns
        if pd.api.types.is_string_dtype(data[column])
    ]


def _uploaded_bytes(uploaded_file: BinaryIO) -> bytes:
    declared_size = getattr(uploaded_file, "size", None)
    if isinstance(declared_size, int) and declared_size > MAX_CSV_BYTES:
        raise CsvValidationError("CSV is too large. Upload a file up to 10 MB.")

    getvalue = getattr(uploaded_file, "getvalue", None)
    if callable(getvalue):
        payload = getvalue()
    else:
        payload = uploaded_file.read(MAX_CSV_BYTES + 1)

    if not isinstance(payload, bytes):
        raise CsvValidationError("CSV reading failed. Upload a UTF-8 CSV file.")
    if len(payload) > MAX_CSV_BYTES:
        raise CsvValidationError("CSV is too large. Upload a file up to 10 MB.")
    return payload


def read_tweet_csv(uploaded_file: BinaryIO) -> pd.DataFrame:
    payload = _uploaded_bytes(uploaded_file)
    if not payload:
        raise CsvValidationError("CSV is empty. Upload a CSV with tweet text.")

    try:
        data = pd.read_csv(io.BytesIO(payload), nrows=MAX_CSV_ROWS + 1)
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
    ) as error:
        raise CsvValidationError(
            "CSV reading failed. Upload a valid UTF-8 CSV file."
        ) from error

    if len(data) > MAX_CSV_ROWS:
        raise CsvValidationError(
            f"CSV has too many rows. Upload up to {MAX_CSV_ROWS} rows."
        )
    return data


def predict_texts(
    texts: Iterable[str],
    model: Any,
    vectorizer: Any,
    cleaner: Callable[[str], str] = clean_text,
) -> pd.DataFrame:
    original_texts = list(texts)
    cleaned_texts = [cleaner(text) for text in original_texts]
    vectors = vectorizer.transform(cleaned_texts)
    predictions = model.predict(vectors)
    probabilities = model.predict_proba(vectors)

    try:
        sentiments = [SENTIMENT_LABELS[int(value)] for value in predictions]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Model returned an unsupported sentiment class.") from error
    confidences = np.max(probabilities, axis=1)
    return pd.DataFrame(
        {
            "tweet": original_texts,
            "sentiment": sentiments,
            "confidence": np.round(confidences, 4),
        }
    )


def neutralize_spreadsheet_formula(value: object) -> object:
    if isinstance(value, str) and SPREADSHEET_FORMULA_PATTERN.search(value):
        return f"'{value}"
    return value


def predictions_csv(results: pd.DataFrame) -> bytes:
    safe_results = results.copy()
    safe_results["tweet"] = safe_results["tweet"].map(neutralize_spreadsheet_formula)
    return safe_results.to_csv(index=False).encode("utf-8")


def _show_prediction(sentiment: str, confidence: float) -> None:
    st.progress(confidence)
    st.write(f"Confidence: {confidence * 100:.2f}%")

    message = f"Model predicts the tweet is: {sentiment}"
    if sentiment == "neutral":
        st.info(f"{message} 😶")
    elif sentiment == "positive":
        st.success(f"{message} 🙂")
    else:
        st.error(f"{message} 😞")


def _prediction_error(error: Exception) -> None:
    if isinstance(error, LookupError):
        st.error("NLTK data is missing. Install the resources from the README.")
    else:
        st.error("Prediction failed. Confirm the model files and input.")


def run_app() -> None:
    st.title("Tweet Sentiment App")
    tweet = st.text_area(
        "Enter your tweet here!",
        placeholder=(
            "I am really frustrated with this service. "
            "It keeps crashing and nothing works."
        ),
        height=160,
    )

    if st.button("Predict"):
        if not tweet.strip():
            st.warning("Enter a tweet first.")
        else:
            try:
                model, vectorizer = load_artifacts()
                result = predict_texts([tweet], model, vectorizer).iloc[0]
                _show_prediction(str(result["sentiment"]), float(result["confidence"]))
            except (LookupError, OSError, ValueError) as error:
                _prediction_error(error)

    st.subheader("Batch CSV Prediction")
    st.caption("Upload an Xquik tweet export or another CSV with tweet text.")
    uploaded_csv = st.file_uploader("Tweet CSV", type=["csv"])
    if uploaded_csv is None:
        return

    try:
        csv_data = read_tweet_csv(uploaded_csv)
    except CsvValidationError as error:
        st.warning(str(error))
        return

    if csv_data.empty:
        st.warning("CSV has no data rows. Upload a CSV with tweet text.")
        return

    text_columns = find_text_columns(csv_data)
    if not text_columns:
        st.warning("No text column found. Add a Tweet Text or text column.")
        return

    text_column = st.selectbox("Tweet text column", text_columns)
    max_rows = min(MAX_CSV_ROWS, len(csv_data))
    rows_to_analyze = st.number_input(
        "Rows to analyze",
        min_value=1,
        max_value=max_rows,
        value=min(100, max_rows),
    )
    if not st.button("Predict CSV", key="predict_csv"):
        return

    texts = csv_data[text_column].dropna().astype(str).str.strip()
    texts = texts[texts != ""].head(int(rows_to_analyze))
    if texts.empty:
        st.warning("No non-empty tweets found in the selected column.")
        return

    try:
        model, vectorizer = load_artifacts()
        results = predict_texts(texts, model, vectorizer)
    except (LookupError, OSError, ValueError) as error:
        _prediction_error(error)
        return

    st.dataframe(results, use_container_width=True)
    st.download_button(
        "Download Predictions CSV",
        predictions_csv(results),
        "tweet_sentiment_predictions.csv",
        "text/csv",
    )


if __name__ == "__main__":
    run_app()
