# Interview Preparation – Spam Email Detection

## 1-minute explanation
"I built a spam detection system that classifies a message as spam or not spam. I used the UCI SMS Spam Collection dataset, cleaned it, split it into training and testing sets with stratification, converted text into numbers with TF-IDF, and trained a Multinomial Naive Bayes classifier inside a scikit-learn Pipeline. I evaluated it with accuracy, precision, recall, F1-score and a confusion matrix, saved the pipeline with joblib, and built a Streamlit app where users paste a message and see the prediction and a probability estimate. I also wrote pytest tests. It's a baseline model trained on SMS data, so it isn't a full email security system."

## Problem and solution
**Problem:** automatically decide whether a message is spam. **Solution:** supervised text classification – learn word patterns from labeled examples, then apply them to new messages.

## Key concepts (simple answers)
- **Why Naive Bayes?** It is fast, simple, works well on small text datasets with many features, and is easy to explain. It's a strong baseline.
- **NLP:** Natural Language Processing – getting computers to work with human language. Here, turning text into numbers a model can use.
- **TF-IDF:** Term Frequency × Inverse Document Frequency. A word scores high if it appears often in a message but rarely across all messages, so common words ("the") get low weight and distinctive words ("prize") get high weight.
- **Multinomial Naive Bayes:** uses Bayes' theorem to compute the probability of each class given the words. "Naive" = assumes words are independent of each other. "Multinomial" = suited to counts/frequency-like features such as word counts or TF-IDF.
- **Training vs testing data:** training data is what the model learns from; testing data is unseen data used once to measure how well it generalises.
- **Why stratified splitting?** Spam is the minority class. Stratifying keeps the same spam/ham ratio in both sets, so the test set is representative.
- **Accuracy:** share of all predictions that are correct. **Precision:** of messages predicted spam, how many are really spam. **Recall:** of all real spam, how many we caught. **F1:** the harmonic mean of precision and recall. With imbalanced data, accuracy alone can look good even if many spam messages are missed.
- **Confusion matrix:** a table of actual vs predicted: true negatives, false positives (ham flagged as spam), false negatives (spam missed), true positives.
- **Data leakage:** information from the test set influencing training (e.g. fitting TF-IDF on all data). It makes results look better than reality. A Pipeline fitted only on training data prevents it.
- **joblib serialization:** saving a trained Python object to a file (`.joblib`) so the app can load it later without retraining.
- **Limitations:** SMS not email, English only, small/dated data, bag-of-words ignores context, estimated probabilities can be over-confident, no attachment/link/sender checks.

## 18 likely interview questions
1. **What is supervised learning?** Learning from labeled examples (message + spam/ham label).
2. **Is this classification or regression?** Classification – predicts a category.
3. **Why do we convert text to numbers?** ML models work on numbers, not raw text.
4. **TF-IDF vs CountVectorizer?** Count just counts words; TF-IDF also down-weights words common in every message.
5. **What are stop words?** Very common words (the, is, and) that carry little meaning; `stop_words="english"` removes them.
6. **Why use a Pipeline?** It bundles TF-IDF and the classifier so the same steps run in training and prediction, and leakage is avoided.
7. **Why `random_state=42`?** To get the same split every run (reproducibility).
8. **What does `stratify` do?** Keeps class proportions the same in train and test.
9. **Why not just use accuracy?** If most messages are ham, a model that always says "ham" scores high accuracy but catches no spam.
10. **Which is worse: false positive or false negative?** Depends on the use case. A false positive hides a real message; a false negative lets spam through. Many systems favour high precision so legitimate mail isn't lost.
11. **What does `predict_proba` return?** The model's estimated probability for each class.
12. **Is a 95% probability a guarantee?** No, it's an estimate; Naive Bayes is often over-confident.
13. **What is overfitting?** Memorising training data and failing on new data. We check by evaluating on a held-out test set.
14. **How would you improve the model?** Cross-validation, tuning (on training data only), n-grams, other models (Logistic Regression, SVM), more/better email data.
15. **Why remove duplicates?** Identical messages in train and test would inflate the score (a form of leakage).
16. **Why Streamlit?** Turns Python scripts into a web UI quickly without front-end code.
17. **Why separate train and predict code?** Training is run once and is slow; prediction runs often and only loads the saved model.
18. **Can it be used for real email?** Only as a baseline. Real email needs email-specific data and features (headers, links, sender reputation, attachments).

## Line-by-line explanation of the important code

### `predict.py`
- `MODEL_PATH` (from `spam_utils`): absolute path of the saved pipeline, so it works from any folder.
- `_cached_model = None`: remembers the loaded model so we read the file only once.
- `load_model()`: checks the file exists; if not, raises `FileNotFoundError` with instructions; otherwise `joblib.load` restores the pipeline.
- `get_model()`: loads the model the first time, then reuses it.
- `predict_message(text, model=None)`:
  - `if not isinstance(text, str) or not text.strip()`: rejects non-text or empty input with `ValueError`.
  - `model.predict([cleaned_text])[0]`: the pipeline applies TF-IDF then Naive Bayes; the input is a list because scikit-learn expects many samples.
  - `class_order = list(model.classes_)`: order of the probability columns (alphabetical: `ham`, `spam`).
  - `probabilities.index("spam")`: picks the spam probability from the right column.
  - Returns a dict with the label and spam probability.
- `if __name__ == "__main__"`: lets you run `python predict.py "text"` from the terminal.

### `train_model.py`
- `matplotlib.use("Agg")`: draw charts to files without opening windows.
- `RANDOM_SEED`, `TEST_SIZE`: constants for a repeatable 80/20 split.
- `explore_data()`: prints total and per-class counts and saves a bar chart.
- `build_pipeline()`: `TfidfVectorizer` (lowercases, removes English stop words) followed by `MultinomialNB`.
- `evaluate_model()`: predicts on the test set and calculates accuracy, precision, recall, F1 (`pos_label="spam"` treats spam as positive) and the confusion matrix.
- `save_confusion_matrix_plot()`: Seaborn heatmap saved to `reports/`.
- `main()`:
  - `load_dataset()` inside `try/except`: prints a friendly error if data is missing.
  - `train_test_split(..., stratify=data["label"])`: split **before** fitting anything.
  - `model.fit(x_train, y_train)`: TF-IDF learns vocabulary from training text only, then Naive Bayes trains.
  - `evaluate_model(model, x_test, y_test)`: the test set is used only here.
  - `joblib.dump(model, MODEL_PATH)`: saves the whole pipeline; `metrics.json` stores real results for the app.

### `app.py`
- `st.set_page_config`: page title/icon and centered layout.
- `SAMPLE_MESSAGES`: example texts for demos.
- `st.session_state["message_text"]`: Streamlit reruns the script on every click; session state keeps the text between runs.
- `use_sample()`: callback that fills the text box when a sample button is clicked.
- `try: model = get_model() except FileNotFoundError`: shows instructions and `st.stop()` if the model isn't trained.
- `st.text_area(..., key="message_text")`: input box bound to session state.
- `if st.button("Predict")`: empty input → `st.warning`; otherwise call `predict_message`.
- Shows spam (red) or not spam (green) plus a confidence value; for "ham", confidence = 1 − spam probability. A caption explains it's an estimate.
- Performance section: reads `reports/metrics.json` and shows the saved confusion matrix image – values come from the real training run, never hard-coded.

### `spam_utils.py` (helper)
- Builds paths from `Path(__file__)`.
- `load_dataset()` finds the file, detects tab/comma separator, tries UTF-8 then latin-1, finds label/text columns by common names, lowercases labels, drops empty rows and duplicate messages.
