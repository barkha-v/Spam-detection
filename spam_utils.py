"""
Shared helpers used by train_model.py, predict.py and app.py.

Why this file exists: both training and prediction need the same file paths
and the same dataset-loading rules. Keeping them in one place avoids copy-paste
mistakes (and means training and prediction can never disagree).
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# File paths. They are built from this file's location, so the project works
# no matter which folder you launch the scripts from.
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

MODEL_PATH = MODELS_DIR / "spam_pipeline.joblib"      # TF-IDF + Naive Bayes
METRICS_PATH = REPORTS_DIR / "metrics.json"           # real evaluation numbers
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
CLASS_BALANCE_PATH = REPORTS_DIR / "class_distribution.png"

# Files we look for inside data/, in this order.
CANDIDATE_DATA_FILES = ["SMSSpamCollection", "spam.csv", "sms.tsv", "spam.tsv", "sms_spam.csv"]

# Column names seen in different versions of this dataset.
LABEL_COLUMN_NAMES = ["label", "v1", "category", "class", "type"]
TEXT_COLUMN_NAMES = ["message", "v2", "text", "sms", "email", "body"]

LABELS = ("ham", "spam")  # the only two labels we use


def find_dataset_file():
    """Return the path of the first dataset file found in data/, or None."""
    for file_name in CANDIDATE_DATA_FILES:
        candidate = DATA_DIR / file_name
        if candidate.exists():
            return candidate
    return None


def _read_raw_file(file_path):
    """Read a dataset file, trying both common encodings and separators."""
    # Some versions of the dataset are not valid UTF-8, so fall back to latin-1.
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(file_path, encoding=encoding) as file:
                first_line = file.readline()
            # The official UCI file is tab-separated; Kaggle's spam.csv uses commas.
            if "\t" in first_line:
                # quoting=3 means "treat quote marks as normal text" (SMS contain many).
                return pd.read_csv(file_path, sep="\t", header=None, encoding=encoding,
                                   quoting=3, dtype=str, keep_default_na=False)
            return pd.read_csv(file_path, sep=",", header=None, encoding=encoding,
                               dtype=str, keep_default_na=False, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {file_path} as UTF-8 or latin-1 text.")


def _standardise_columns(raw):
    """Turn a raw table into a DataFrame with exactly two columns: label, message."""
    first_row = [str(value).strip().lower() for value in raw.iloc[0].tolist()]
    has_header = any(name in LABEL_COLUMN_NAMES for name in first_row)

    if has_header:
        # First row holds column names (e.g. Kaggle's spam.csv: v1, v2, ...).
        raw.columns = first_row
        raw = raw.iloc[1:]
        label_col = next((c for c in raw.columns if c in LABEL_COLUMN_NAMES), None)
        text_col = next((c for c in raw.columns if c in TEXT_COLUMN_NAMES), None)
        if label_col is None or text_col is None:
            raise ValueError(f"Could not find label/text columns. Columns found: {list(raw.columns)}")
        data = raw[[label_col, text_col]].copy()
    else:
        # No header (official UCI file): column 0 = label, column 1 = message.
        data = raw.iloc[:, :2].copy()

    data.columns = ["label", "message"]
    return data


def load_dataset(file_path=None):
    """
    Load and clean the dataset. Returns a DataFrame with columns: label, message.

    Cleaning steps:
      1. Drop rows with missing values.
      2. Make labels consistent (lowercase 'spam' / 'ham').
      3. Remove empty messages and duplicate messages.
    """
    if file_path is None:
        file_path = find_dataset_file()
    if file_path is None or not Path(file_path).exists():
        raise FileNotFoundError(
            "Dataset not found in the data/ folder.\n"
            "Run this first:  python download_data.py\n"
            "See data/README.md for manual setup."
        )

    data = _standardise_columns(_read_raw_file(file_path))

    data = data.dropna()
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data["message"] = data["message"].astype(str).str.strip()

    # Some datasets use 'not spam' or 0/1 - map them to our two labels.
    data["label"] = data["label"].replace({"not spam": "ham", "0": "ham", "1": "spam"})

    data = data[data["label"].isin(LABELS)]       # keep only ham/spam rows
    data = data[data["message"] != ""]            # remove empty messages
    data = data.drop_duplicates(subset="message") # remove duplicate messages
    data = data.reset_index(drop=True)

    if data.empty:
        raise ValueError("The dataset has no usable rows after cleaning.")
    return data
