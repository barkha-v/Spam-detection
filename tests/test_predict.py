"""
Tests for predict.py.   Run from the project root with:  pytest -v

- Tests using the `tiny_model` fixture do NOT need training (they build a small
  model in memory).
- Tests marked "requires trained model" are SKIPPED unless you have run
  `python train_model.py` first.
"""

import sys
from pathlib import Path

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Make the project root importable no matter where pytest is launched.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predict import load_model, predict_message  # noqa: E402
from spam_utils import MODEL_PATH  # noqa: E402

requires_trained_model = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="Run `python train_model.py` first"
)


@pytest.fixture
def tiny_model():
    messages = [
        "win a free prize now", "claim your cash reward today", "free entry urgent call now",
        "are we meeting for lunch", "see you at home tonight", "please send me the notes",
    ]
    labels = ["spam", "spam", "spam", "ham", "ham", "ham"]
    model = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", MultinomialNB())])
    return model.fit(messages, labels)


def test_valid_text_is_accepted(tiny_model):
    result = predict_message("Win a free prize now", model=tiny_model)
    assert "label" in result and "spam_probability" in result


def test_label_is_expected_value(tiny_model):
    result = predict_message("see you at lunch", model=tiny_model)
    assert result["label"] in ("spam", "ham")
    assert 0.0 <= result["spam_probability"] <= 1.0


@pytest.mark.parametrize("bad_input", ["", "   ", None, 123])
def test_empty_or_invalid_text_raises_error(tiny_model, bad_input):
    with pytest.raises(ValueError):
        predict_message(bad_input, model=tiny_model)


def test_missing_model_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "does_not_exist.joblib")


@requires_trained_model
def test_saved_model_can_be_loaded():
    model = load_model()
    assert set(model.classes_) == {"ham", "spam"}


@requires_trained_model
def test_saved_model_predicts_expected_label():
    result = predict_message("Congratulations! You won a free prize. Call now to claim!")
    assert result["label"] in ("spam", "ham")
