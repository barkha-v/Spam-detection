"""
Prediction code (kept separate from training).

Usage from the command line:
    python predict.py "Congratulations! You won a free prize, call now"
"""

import sys

import joblib

from spam_utils import MODEL_PATH

_cached_model = None  # load the model from disk only once


def load_model(model_path=MODEL_PATH):
    """Load the saved pipeline. Raises FileNotFoundError if not trained yet."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}.\n"
            "Train it first:  python train_model.py"
        )
    return joblib.load(model_path)


def get_model():
    global _cached_model
    if _cached_model is None:
        _cached_model = load_model()
    return _cached_model


def predict_message(text, model=None):
    """
    Classify one message.

    Returns a dict: {"label": "spam" or "ham", "spam_probability": float}
    Raises ValueError for empty / non-text input.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter a non-empty message.")

    if model is None:
        model = get_model()

    cleaned_text = text.strip()
    label = model.predict([cleaned_text])[0]

    # predict_proba gives one probability per class, ordered like model.classes_
    class_order = list(model.classes_)
    probabilities = model.predict_proba([cleaned_text])[0]
    spam_probability = float(probabilities[class_order.index("spam")])

    return {"label": str(label), "spam_probability": spam_probability}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python predict.py "your message here"')
        raise SystemExit(1)
    try:
        result = predict_message(" ".join(sys.argv[1:]))
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"Prediction: {result['label']}  (spam probability: {result['spam_probability']:.2%})")
