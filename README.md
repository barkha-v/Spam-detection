# Spam Email Detection System

A beginner-friendly Machine Learning project that classifies a message as **spam** or **not spam (ham)** using Natural Language Processing (TF-IDF) and a Multinomial Naive Bayes classifier, with a Streamlit web interface.

## Overview
The app learns from the labeled SMS Spam Collection dataset and then predicts whether newly entered text looks like spam. It is a **baseline text classifier**, built to demonstrate a clean end-to-end ML workflow: data cleaning, training, evaluation, saving the model, a web app, and tests.

## Problem statement
Unwanted spam messages waste time and can be used for scams. Given the text of a message, decide automatically whether it is spam or legitimate.

## Objectives
- Build a complete ML pipeline using NLP and Naive Bayes.
- Evaluate the model honestly on data it has never seen.
- Provide an easy web interface to try the model.
- Keep the code simple enough to explain in an interview.

## Features
- Dataset download/setup script and tolerant loader (handles common CSV/TSV formats, missing values, duplicates, label cleanup).
- Stratified train/test split with a fixed random seed.
- Single scikit-learn `Pipeline` (TF-IDF + Multinomial Naive Bayes) – no data leakage.
- Metrics: accuracy, precision, recall, F1-score, confusion matrix (saved to `reports/`).
- Streamlit app: text box, sample messages, Predict button, probability estimate, empty-input warning, real metrics and confusion matrix.
- Pytest tests, and a helpful error if the model has not been trained yet.

## Technologies used
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Streamlit · Joblib · Pytest

## Machine Learning workflow
1. Load the dataset → 2. Explore (spam vs ham counts) → 3. Clean (missing values, duplicates, labels) → 4. Stratified 80/20 split (seed 42) → 5. TF-IDF features (fitted on training data only) → 6. Train Multinomial Naive Bayes → 7. Predict on the test set → 8. Evaluate → 9. Save the whole pipeline with joblib → 10. Load it in `predict.py` and `app.py`.

The test set is used only once, for the final evaluation. It is never used for tuning.

## Project folder structure
```
spam-email-detection/
├── app.py                  # Streamlit web app
├── train_model.py          # trains, evaluates, saves model + reports
├── predict.py              # loads model and predicts (CLI + function)
├── spam_utils.py           # helper: file paths + dataset loading/cleaning
├── download_data.py        # helper: downloads the UCI dataset
├── requirements.txt
├── README.md
├── INTERVIEW_PREP.md       # interview answers + line-by-line code notes
├── GITHUB_GUIDE.md         # Git/GitHub commands for Windows
├── .gitignore
├── LICENSE
├── data/README.md          # dataset info (data file is downloaded, not committed)
├── models/.gitkeep         # trained model is saved here
├── notebooks/exploratory_analysis.ipynb
├── reports/.gitkeep        # metrics.json and charts are saved here
└── tests/test_predict.py
```
`spam_utils.py` and `download_data.py` are the only additions to the requested structure. `spam_utils.py` keeps paths and data loading in one place so training and prediction always agree; `download_data.py` keeps the dataset out of Git.

## Dataset source and setup
**SMS Spam Collection** from the UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/228/sms+spam+collection (about 5,574 messages, labeled ham/spam). UCI lists it under CC BY 4.0 – please check the page for current terms and credit the authors (Almeida & Hidalgo, 2011). Details are in `data/README.md`.

```
python download_data.py
```
If the download fails, follow the manual steps in `data/README.md`.

Because the data is SMS text, the model is a baseline for spam-*like* messages. It is not a production-grade email security system.

## Installation (Windows)
1. Install Python 3.9+ from https://www.python.org (tick **"Add Python to PATH"**).
2. Open a terminal in the project folder (VS Code: *Terminal → New Terminal*).

### Virtual environment
```
py -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
(PowerShell may block activation. If so, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt.)
On macOS/Linux use `python3 -m venv venv` and `source venv/bin/activate`.

## Model training
```
python download_data.py
python train_model.py
```
This prints the data counts and the evaluation results, and creates `models/spam_pipeline.joblib`, `reports/metrics.json`, `reports/confusion_matrix.png` and `reports/class_distribution.png`.

## Launch the Streamlit app
```
streamlit run app.py
```
Open http://localhost:8501 if the browser doesn't open automatically. Quick command-line check:
```
python predict.py "Congratulations! You won a free prize, call now"
```

## Run tests
```
pytest -v
```
- Tests that use a small in-memory model run **without** training.
- Two tests ("saved model can be loaded", "saved model predicts a valid label") are **skipped** until you have run `python train_model.py`.

## Example messages
| Type | Message |
|------|---------|
| Spam | `WINNER!! You have been selected to receive a free prize. Call 09061701461 now to claim!` |
| Spam | `URGENT! Your mobile number has won a 2000 cash award. Text CLAIM to 81010.` |
| Ham | `Hey, are we still meeting for lunch at 1pm tomorrow?` |
| Ham | `Can you send me the notes from today's class?` |

## Evaluation metrics and reproducing results
This README deliberately lists **no numbers**: your results come from running the code on your machine. After `python train_model.py`, read them from the terminal, from `reports/metrics.json`, or from the app's *Model performance* section. The split uses a fixed seed (42) so the same dataset gives the same split. Small differences can still appear with different library versions.

Spam is treated as the positive class: **precision** = of messages flagged spam, how many really were; **recall** = of real spam, how many were caught.

### Adding screenshots
Run the app, take a screenshot (Windows: `Win + Shift + S`), save it in a new `screenshots/` folder, and add `![App screenshot](screenshots/app.png)` to this README.

## Limitations
- Trained on SMS messages, not full emails (no headers, HTML, attachments or links analysis).
- Bag-of-words ignores word order and context.
- The dataset is small, English only and dated; modern spam differs.
- Class imbalance (far more ham than spam) means accuracy alone can mislead – check recall and F1.
- Probabilities from Naive Bayes are estimates and often over-confident.
- The app cannot verify senders, scan attachments or access an inbox.

## Future enhancements
Cross-validation and hyperparameter tuning (on training data only), n-gram features, comparison with Logistic Regression/SVM, an email-specific dataset, simple text cleaning options, Docker deployment, a CI workflow for tests.

## License and attribution
Code: MIT License (see `LICENSE`). Dataset: SMS Spam Collection (Almeida & Hidalgo, 2011), UCI Machine Learning Repository – see `data/README.md`.

## Contributor
**barkha-v** – https://github.com/barkha-v

## Common errors and fixes
| Error | Fix |
|-------|-----|
| `Dataset not found in the data/ folder` | Run `python download_data.py` or follow `data/README.md`. |
| `Trained model not found` / app shows red error | Run `python train_model.py` first. |
| `'streamlit' is not recognized` | Activate the venv (`venv\Scripts\activate`) and run `pip install -r requirements.txt`. |
| `ModuleNotFoundError: spam_utils` | Run commands from the project root folder. |
| Activation blocked in PowerShell | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. |
| Download fails (firewall/offline) | Download manually from UCI; see `data/README.md`. |
| Port 8501 busy | `streamlit run app.py --server.port 8502`. |
