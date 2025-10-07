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

### Train DistilBERT (alternative, higher-quality model)
```bash
pip install -r requirements.txt  # ensure transformers/torch installed
python3 train_distilbert.py --csv hotel_reviews.csv --model-name distilbert-base-uncased --epochs 3 --batch-size 16
```
- HF model saved to `artifacts/hotel_sentiment_distilbert/` (includes tokenizer + model)
- Reports saved to `reports/` (classification report, confusion matrix)

### Run the app (single text input)
```bash
python3 -m streamlit run app.py
```

### Files
- `train.py` — trains and evaluates the TF-IDF + Logistic Regression model
- `train_distilbert.py` — fine-tunes DistilBERT for 3-class sentiment
- `app.py` — Streamlit app for single review prediction (auto-loads HF model if present)
- `hotel_reviews.csv` — data (needs columns: `Review`, `Rating`)
- `reports/` — evaluation outputs
- `artifacts/` — saved model
- `requirements.txt` — dependencies

### Label mapping
- Ratings 1–2 → Negative
- Rating 3 → Neutral
- Ratings 4–5 → Positive


