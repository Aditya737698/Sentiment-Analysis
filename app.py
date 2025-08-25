import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "artifacts/hotel_sentiment_pipeline.joblib"
CLASS_ORDER = ["Negative", "Neutral", "Positive"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict_single(model, text: str):
    if not text or not text.strip():
        return None, None
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba([text])[0]
        except Exception:
            proba = None
    pred = model.predict([text])[0]
    return pred, proba


st.title("Hotel Review Sentiment Analysis")
st.caption("TF-IDF + Logistic Regression pipeline with evaluation artifacts.")

model = load_model()

review = st.text_area("Enter a hotel review:")
if st.button("Predict"):
    label, proba = predict_single(model, review)
    if label is None:
        st.warning("Please enter a non-empty review.")
    else:
        st.subheader(f"Predicted Sentiment: {label}")
        if proba is not None:
            proba_series = pd.Series(proba, index=getattr(model, "classes_", CLASS_ORDER))
            st.bar_chart(proba_series)


