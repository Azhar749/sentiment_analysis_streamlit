import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Sentiment Analysis", page_icon="😊", layout="centered")

EMOJI_MAP = {
    "positive": "✅ Positive",
    "neutral": "⚪ Neutral",
    "negative": "❌ Negative",
}

EXPLANATION_MAP = {
    "positive": "This feedback is positive and shows customer satisfaction.",
    "neutral": "This feedback is neutral and does not express strong sentiment.",
    "negative": "This feedback is negative and indicates customer dissatisfaction.",
}

SAMPLE_DATA = [
    ("I loved the service and the support team was amazing.", "positive"),
    ("The product quality is outstanding and I would recommend it.", "positive"),
    ("This was a great experience and I am very happy.", "positive"),
    ("I am not impressed, but the delivery was fine.", "neutral"),
    ("The app is okay, not too good and not too bad.", "neutral"),
    ("It works, though I expected a few more features.", "neutral"),
    ("The support response was slow and the product keeps crashing.", "negative"),
    ("I am disappointed with the service and will not use it again.", "negative"),
    ("The experience was frustrating and the quality was poor.", "negative"),
]

@st.cache_resource
def build_model():
    df = pd.DataFrame(SAMPLE_DATA, columns=["text", "label"])
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    pipeline = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    return pipeline, score


def classify_feedback(model, text):
    prediction = model.predict([text])[0]
    probability = model.predict_proba([text]).max()
    return prediction, probability


def main():
    st.title("Sentiment Analysis App")
    st.write(
        "Analyze customer feedback in real time using a Streamlit-based NLP sentiment classifier. "
        "The app labels text as positive, neutral, or negative and shows a matching reaction emoji."
    )

    with st.expander("How it works"):
        st.write(
            "This app uses a TF-IDF vectorizer and Logistic Regression model built from sample customer feedback. "
            "Enter a sentence and receive an instant sentiment prediction with a friendly emoji classification."
        )
        st.write("### Sample training data")
        st.write(pd.DataFrame(SAMPLE_DATA, columns=["Feedback text", "Sentiment"]))

    model, accuracy = build_model()
    st.metric(label="Model accuracy on sample data", value=f"{accuracy * 100:.1f}%")

    feedback_text = st.text_area("Enter customer feedback here", height=140, placeholder="Type feedback like: 'The service was great, very satisfied.'")

    if st.button("Analyze Feedback"):
        if not feedback_text.strip():
            st.warning("Please enter feedback text to classify.")
        else:
            sentiment, confidence = classify_feedback(model, feedback_text)
            st.success(f"Sentiment: {EMOJI_MAP[sentiment]}")
            st.write(EXPLANATION_MAP[sentiment])
            st.write(f"Confidence: {confidence * 100:.1f}%")

    st.markdown("---")
    st.subheader("Tips")
    st.write(
        "- Use clear customer feedback sentences for better predictions."
        "\n- Positive, neutral, and negative labels are shown with emojis to match sentiment." 
        "\n- This app is built to be ready to run locally with Streamlit."
    )

    st.sidebar.header("Project Info")
    st.sidebar.write("Sentiment analysis project using Streamlit and NLP.")
    st.sidebar.write("Upload your own data later by extending this app.")


if __name__ == "__main__":
    main()
