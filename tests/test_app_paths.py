import inspect
import tempfile
import unittest
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from app_paths import MODEL_DIRECTORY, MODEL_PATH, VECTORIZER_PATH
from model import save_artifacts


class AppPathTests(unittest.TestCase):
    def test_model_artifacts_use_the_committed_model_directory(self) -> None:
        self.assertEqual(MODEL_PATH.parent, MODEL_DIRECTORY)
        self.assertEqual(VECTORIZER_PATH.parent, MODEL_DIRECTORY)
        self.assertTrue(MODEL_PATH.is_file())
        self.assertTrue(VECTORIZER_PATH.is_file())

        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)

        self.assertTrue(callable(model.predict))
        self.assertTrue(callable(vectorizer.transform))

    def test_training_defaults_to_the_committed_artifact_paths(self) -> None:
        parameters = inspect.signature(save_artifacts).parameters

        self.assertEqual(parameters["model_path"].default, MODEL_PATH)
        self.assertEqual(parameters["vectorizer_path"].default, VECTORIZER_PATH)

    def test_training_writes_both_artifacts_to_the_requested_directory(self) -> None:
        vectorizer = TfidfVectorizer()
        model = RandomForestClassifier()

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            vectorizer_path = destination / "model" / "vectorizer.pkl"
            model_path = destination / "model" / "model.pkl"

            save_artifacts(vectorizer, model, vectorizer_path, model_path)

            self.assertIsInstance(joblib.load(vectorizer_path), TfidfVectorizer)
            self.assertIsInstance(joblib.load(model_path), RandomForestClassifier)


if __name__ == "__main__":
    unittest.main()
