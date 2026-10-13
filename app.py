"""
Streamlit web app.   Run with:  streamlit run app.py
"""

import json

import streamlit as st

from predict import get_model, predict_message
from spam_utils import CONFUSION_MATRIX_PATH, METRICS_PATH

st.set_page_config(page_title="Spam Email Detection", page_icon="📧", layout="centered")

SAMPLE_MESSAGES = {
    "Spam example 1": "WINNER!! You have been selected to receive a free prize. Call 09061701461 now to claim!",
    "Spam example 2": "URGENT! Your mobile number has won a 2000 cash award. Text CLAIM to 81010 to collect.",
    "Ham example 1": "Hey, are we still meeting for lunch at 1pm tomorrow?",
    "Ham example 2": "Can you send me the notes from today's class? I missed the last part.",
}

# Session state keeps the text box content when a sample button is clicked.
if "message_text" not in st.session_state:
    st.session_state["message_text"] = ""


def use_sample(text):
    st.session_state["message_text"] = text


st.title("📧 Spam Email Detection System")
st.write(
    "A baseline Machine Learning app that estimates whether a message looks like "
    "**spam** or **not spam (ham)**, using TF-IDF and Multinomial Naive Bayes."
)

# Stop early with a helpful message if the model has not been trained yet.
try:
    model = get_model()
except FileNotFoundError as error:
    st.error("The trained model was not found.")
    st.code("python download_data.py\npython train_model.py")
    st.caption(str(error))
    st.stop()

st.subheader("Try a message")
st.caption("Click a sample to fill the box, or paste your own text.")
sample_columns = st.columns(2)
for index, (name, text) in enumerate(SAMPLE_MESSAGES.items()):
    sample_columns[index % 2].button(name, on_click=use_sample, args=(text,), use_container_width=True)

st.text_area("Message or email text", key="message_text", height=160,
             placeholder="Paste a message here...")

if st.button("Predict", type="primary"):
    user_text = st.session_state["message_text"]
    if not user_text.strip():
        st.warning("Please enter some text before clicking Predict.")
    else:
        result = predict_message(user_text, model=model)
        spam_probability = result["spam_probability"]
        if result["label"] == "spam":
            st.error("🚨 Result: **Spam**")
            confidence = spam_probability
        else:
            st.success("✅ Result: **Not Spam (Ham)**")
            confidence = 1 - spam_probability
        st.metric("Model confidence in this result", f"{confidence:.1%}")
        st.progress(min(max(confidence, 0.0), 1.0))
        st.caption(
            "This percentage is the model's own estimate, not a guarantee. "
            "Always be careful with suspicious messages."
        )

st.divider()
st.subheader("Model performance")
if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    col2.metric("Precision", f"{metrics['precision']:.3f}")
    col3.metric("Recall", f"{metrics['recall']:.3f}")
    col4.metric("F1-score", f"{metrics['f1_score']:.3f}")
    st.caption(f"Measured on {metrics['test_size']} held-out test messages "
               f"(spam is the positive class).")
    if CONFUSION_MATRIX_PATH.exists():
        st.image(str(CONFUSION_MATRIX_PATH), caption="Confusion matrix on the test set")
else:
    st.info("Metrics file not found. Run `python train_model.py` to generate it.")

st.divider()
st.caption(
    "Limitations: trained on SMS messages, so it is a baseline text classifier, not a "
    "production email security system. It does not check attachments, links, sender "
    "identity, or access any inbox."
)
