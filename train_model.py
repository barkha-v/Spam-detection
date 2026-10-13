"""
Train the spam classifier and save it.

Usage:  python train_model.py

Steps: load data -> explore -> split -> TF-IDF + Naive Bayes pipeline ->
       evaluate on the test set -> save model, metrics and charts.
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")  # draw charts to files without opening a window
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from spam_utils import (CLASS_BALANCE_PATH, CONFUSION_MATRIX_PATH, METRICS_PATH,
                        MODEL_PATH, MODELS_DIR, REPORTS_DIR, load_dataset)

RANDOM_SEED = 42   # fixed so results are reproducible
TEST_SIZE = 0.2    # 20% of the data is kept aside for testing


def explore_data(data):
    """Print basic facts about the data and save a class-balance chart."""
    print("\n--- Data exploration ---")
    print(f"Total messages after cleaning: {len(data)}")
    counts = data["label"].value_counts()
    print(counts.to_string())

    plt.figure(figsize=(5, 4))
    sns.countplot(data=data, x="label", order=["ham", "spam"])
    plt.title("Number of ham vs spam messages")
    plt.tight_layout()
    plt.savefig(CLASS_BALANCE_PATH)
    plt.close()
    return {label: int(count) for label, count in counts.items()}


def build_pipeline():
    """TF-IDF turns text into numbers, then Naive Bayes classifies them."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
        ("classifier", MultinomialNB()),
    ])


def evaluate_model(model, x_test, y_test):
    """Compute metrics on the test set. 'spam' is the positive class."""
    predictions = model.predict(x_test)
    matrix = confusion_matrix(y_test, predictions, labels=["ham", "spam"])
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label="spam"),
        "recall": recall_score(y_test, predictions, pos_label="spam"),
        "f1_score": f1_score(y_test, predictions, pos_label="spam"),
        "confusion_matrix": matrix.tolist(),  # [[TN, FP], [FN, TP]]
    }


def save_confusion_matrix_plot(matrix):
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title("Confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()


def main():
    try:
        data = load_dataset()
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1)

    class_counts = explore_data(data)

    # Split BEFORE fitting anything. stratify keeps the spam/ham ratio equal
    # in the train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(
        data["message"], data["label"],
        test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=data["label"],
    )

    # The pipeline fits TF-IDF on the training data only (no data leakage).
    model = build_pipeline()
    model.fit(x_train, y_train)

    results = evaluate_model(model, x_test, y_test)
    results.update({
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "class_counts": class_counts,
        "random_seed": RANDOM_SEED,
    })

    print("\n--- Evaluation on the test set ---")
    for name in ("accuracy", "precision", "recall", "f1_score"):
        print(f"{name:>10}: {results[name]:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:", results["confusion_matrix"])

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(results, indent=2))
    save_confusion_matrix_plot(results["confusion_matrix"])

    print(f"\nSaved model   -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    print(f"Saved charts  -> {REPORTS_DIR}")


if __name__ == "__main__":
    main()
