import os
import re
import time
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Using scikit-learn's default tokenizer/analyzer (no spell checking or custom negation)


def rating_to_sentiment(rating: int) -> str:
    if rating in [1, 2]:
        return "Negative"
    if rating == 3:
        return "Neutral"
    return "Positive"


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Review" not in df.columns or "Rating" not in df.columns:
        raise ValueError("Input CSV must have 'Review' and 'Rating' columns")
    df["Sentiment"] = df["Rating"].apply(rating_to_sentiment)
    df["Review"] = df["Review"].fillna("")
    return df


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=60000,
            min_df=2
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            n_jobs=None,
            class_weight="balanced",
            solver="liblinear"
        ))
    ])


def ensure_dirs():
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("reports", exist_ok=True)


def train_and_evaluate(csv_path: str = "hotel_reviews.csv", sample_n: int = 0):
    ensure_dirs()
    t0 = time.time()
    print("[1/5] Loading data...")
    df = load_data(csv_path)
    if sample_n and 0 < sample_n < len(df):
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        print(f"Using sample of {len(df)} rows for fast run")

    X = df["Review"].astype(str)
    y = df["Sentiment"].astype(str)

    print("[2/5] Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[3/5] Building pipeline...")
    pipeline = build_pipeline()
    print("[4/5] Fitting model...")
    t_fit = time.time()
    pipeline.fit(X_train, y_train)
    print(f"Fit done in {time.time() - t_fit:.1f}s")

    print("[5/5] Evaluating...")
    y_pred = pipeline.predict(X_test)

    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=["Negative", "Neutral", "Positive"])  # fixed label order

    # Save artifacts
    model_path = os.path.join("artifacts", "hotel_sentiment_pipeline.joblib")
    joblib.dump(pipeline, model_path)

    with open(os.path.join("reports", "classification_report.txt"), "w") as f:
        f.write(report)

    np.savetxt(os.path.join("reports", "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

    # Minimal console output
    print("Saved model to:", model_path)
    print("\nClassification Report:\n", report)
    print("Confusion Matrix (rows=true, cols=pred):\n", cm)
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="hotel_reviews.csv", help="Path to input CSV")
    parser.add_argument("--sample-n", type=int, default=0, help="Use only N rows for a quick run")
    args = parser.parse_args()

    train_and_evaluate(csv_path=args.csv, sample_n=args.sample_n)


