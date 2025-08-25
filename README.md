## Hotel Review Sentiment Analysis 

This project trains a sentiment classifier (Negative/Neutral/Positive) on hotel reviews using scikit-learn, and provides a small Streamlit app to predict sentiment for a single review.

### Setup
```bash
pip install -r requirements.txt
```

### Train (creates model + reports)
```bash
python3 train.py
```
- Model saved to `artifacts/hotel_sentiment_pipeline.joblib`
- Reports saved to `reports/` (classification report, confusion matrix)

### Run the app (single text input)
```bash
python3 -m streamlit run app.py
```

### Files
- `train.py` — trains and evaluates the model
- `app.py` — Streamlit app for single review prediction
- `hotel_reviews.csv` — data (needs columns: `Review`, `Rating`)
- `reports/` — evaluation outputs
- `artifacts/` — saved model
- `requirements.txt` — dependencies

### Label mapping
- Ratings 1–2 → Negative
- Rating 3 → Neutral
- Ratings 4–5 → Positive


