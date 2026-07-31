import io
import unittest

import numpy as np
import pandas as pd

from web_app import (
    MAX_CSV_BYTES,
    MAX_CSV_ROWS,
    CsvValidationError,
    find_text_columns,
    neutralize_spreadsheet_formula,
    predict_texts,
    predictions_csv,
    read_tweet_csv,
)


class FakeVectorizer:
    def __init__(self) -> None:
        self.transformed: list[str] = []

    def transform(self, texts: list[str]) -> list[str]:
        self.transformed = texts
        return texts


class OversizedUpload(io.BytesIO):
    size = MAX_CSV_BYTES + 1

    def getvalue(self) -> bytes:
        raise AssertionError("Oversized uploads must be rejected before reading.")


class FakeModel:
    def __init__(
        self,
        predictions: tuple[int, ...] = (1, 2),
        probabilities: tuple[tuple[float, ...], ...] = (
            (0.1, 0.8, 0.1),
            (0.1, 0.2, 0.7),
        ),
    ) -> None:
        self.predictions = predictions
        self.probabilities = probabilities

    def predict(self, _vectors: list[str]) -> np.ndarray:
        return np.array(self.predictions)

    def predict_proba(self, _vectors: list[str]) -> np.ndarray:
        return np.array(self.probabilities)


class CsvTests(unittest.TestCase):
    def test_xquik_tweet_text_column_is_preferred(self) -> None:
        data = pd.DataFrame(
            {
                "User ID": ["1"],
                "Username": ["person"],
                "Tweet Text": ["A tweet"],
            }
        )

        self.assertEqual(find_text_columns(data), ["Tweet Text"])

    def test_underscored_text_column_is_recognized(self) -> None:
        data = pd.DataFrame({"metadata": ["value"], "tweet_text": ["A tweet"]})

        self.assertEqual(find_text_columns(data), ["tweet_text"])

    def test_string_columns_are_fallbacks(self) -> None:
        data = pd.DataFrame({"Identifier": [1], "Body": ["A tweet"]})

        self.assertEqual(find_text_columns(data), ["Body"])

    def test_read_tweet_csv_accepts_xquik_shape(self) -> None:
        uploaded = io.BytesIO(
            b"User ID,Username,Tweet Text\n1,person,A reliable export\n"
        )

        data = read_tweet_csv(uploaded)

        self.assertEqual(list(data.columns), ["User ID", "Username", "Tweet Text"])
        self.assertEqual(data.iloc[0]["Tweet Text"], "A reliable export")

    def test_read_tweet_csv_rejects_empty_files(self) -> None:
        with self.assertRaisesRegex(CsvValidationError, "CSV is empty"):
            read_tweet_csv(io.BytesIO())

    def test_read_tweet_csv_rejects_large_files(self) -> None:
        uploaded = io.BytesIO(b"x" * (MAX_CSV_BYTES + 1))

        with self.assertRaisesRegex(CsvValidationError, "up to 10 MB"):
            read_tweet_csv(uploaded)

    def test_read_tweet_csv_rejects_declared_large_files_before_reading(self) -> None:
        with self.assertRaisesRegex(CsvValidationError, "up to 10 MB"):
            read_tweet_csv(OversizedUpload())

    def test_read_tweet_csv_rejects_too_many_rows(self) -> None:
        rows = b"text\n" + b"\n".join(b"tweet" for _ in range(MAX_CSV_ROWS + 1))

        with self.assertRaisesRegex(CsvValidationError, "up to 500 rows"):
            read_tweet_csv(io.BytesIO(rows))

    def test_batch_prediction_vectorizes_once(self) -> None:
        vectorizer = FakeVectorizer()

        results = predict_texts(
            ["Great!", "Awful!"],
            FakeModel(),
            vectorizer,
            cleaner=str.lower,
        )

        self.assertEqual(vectorizer.transformed, ["great!", "awful!"])
        self.assertEqual(results["sentiment"].tolist(), ["positive", "negative"])
        self.assertEqual(results["confidence"].tolist(), [0.8, 0.7])

    def test_batch_prediction_rejects_an_unknown_model_class(self) -> None:
        model = FakeModel(predictions=(9,), probabilities=((0.1, 0.2, 0.7),))

        with self.assertRaisesRegex(ValueError, "unsupported sentiment class"):
            predict_texts(
                ["Unexpected"],
                model,
                FakeVectorizer(),
                cleaner=str.lower,
            )

    def test_spreadsheet_formulas_are_neutralized(self) -> None:
        unsafe_values = ["=1+1", "+cmd", "-2+3", "@SUM(A1)", "  =1", "\ttab"]

        for value in unsafe_values:
            with self.subTest(value=value):
                self.assertEqual(neutralize_spreadsheet_formula(value), f"'{value}")

        self.assertEqual(
            neutralize_spreadsheet_formula("ordinary tweet"), "ordinary tweet"
        )

    def test_download_neutralizes_only_the_export_copy(self) -> None:
        results = pd.DataFrame(
            {"tweet": ["=1+1"], "sentiment": ["neutral"], "confidence": [0.9]}
        )

        exported = predictions_csv(results).decode("utf-8")

        self.assertIn("'=1+1", exported)
        self.assertEqual(results.iloc[0]["tweet"], "=1+1")


if __name__ == "__main__":
    unittest.main()
