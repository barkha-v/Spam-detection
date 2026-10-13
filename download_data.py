"""
Download the SMS Spam Collection dataset from the UCI repository.

Usage:  python download_data.py
"""

import io
import sys
import urllib.request
import zipfile

from spam_utils import DATA_DIR

DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
TARGET_FILE_NAME = "SMSSpamCollection"


def download_dataset():
    DATA_DIR.mkdir(exist_ok=True)
    target_path = DATA_DIR / TARGET_FILE_NAME

    if target_path.exists():
        print(f"Dataset already exists at {target_path}. Nothing to do.")
        return

    print(f"Downloading dataset from {DATASET_URL} ...")
    try:
        with urllib.request.urlopen(DATASET_URL, timeout=60) as response:
            zip_bytes = response.read()
    except Exception as error:  # network problems, blocked site, etc.
        print(f"Download failed: {error}")
        print("Please follow the manual setup steps in data/README.md")
        sys.exit(1)

    # The zip contains the data file plus a readme; we only need the data file.
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        if TARGET_FILE_NAME not in archive.namelist():
            print(f"'{TARGET_FILE_NAME}' not found in the zip. Files: {archive.namelist()}")
            sys.exit(1)
        target_path.write_bytes(archive.read(TARGET_FILE_NAME))

    print(f"Saved dataset to {target_path}")


if __name__ == "__main__":
    download_dataset()
