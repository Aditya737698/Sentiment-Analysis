import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import os
from typing import Tuple, Optional

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

MODEL_PATH = "artifacts/hotel_sentiment_pipeline.joblib"
HF_MODEL_DIR = "artifacts/hotel_sentiment_distilbert"
CLASS_ORDER = ["Negative", "Neutral", "Positive"]


@st.cache_resource
def load_model():
    if HF_AVAILABLE and os.path.isdir(HF_MODEL_DIR):
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_DIR)
        model.eval()
        return (tokenizer, model)
    # fallback to sklearn pipeline
    return joblib.load(MODEL_PATH)


def predict_single(model_obj, text: str):
    if not text or not text.strip():
        return None, None
    # HF path
    if HF_AVAILABLE and isinstance(model_obj, tuple):
        tokenizer, hf_model = model_obj
        with torch.no_grad():
            inputs = tokenizer([text], return_tensors="pt", truncation=True)
            outputs = hf_model(**inputs)
            logits = outputs.logits
            proba = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_idx = int(np.argmax(proba))
            return CLASS_ORDER[pred_idx], proba
    # sklearn path
    model = model_obj
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba([text])[0]
        except Exception:
            proba = None
    pred = model.predict([text])[0]
    return pred, proba


st.title("Hotel Review Sentiment Analysis")
if HF_AVAILABLE and os.path.isdir(HF_MODEL_DIR):
    st.caption("DistilBERT (transformers) model with evaluation artifacts.")
else:
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


