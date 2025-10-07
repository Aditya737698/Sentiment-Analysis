# %%
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import joblib



# %%
df = pd.read_csv("hotel_reviews.csv")

def rating_to_sentiment(rating):
    if rating in [1, 2]:
        return 'Negative'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Positive'

df['Sentiment'] = df['Rating'].apply(rating_to_sentiment)

df['clean_text'] = df['Review'].apply(lambda x: re.sub("<.*?>", "", x))
df['clean_text'] = df['clean_text'].apply(lambda x: re.sub('[^\w\s]', "", x))
df['clean_text'] = df['clean_text'].str.lower()



# %%
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stemmer = PorterStemmer()
lemma = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

df['tokenize_text'] = df['clean_text'].apply(lambda y: word_tokenize(y))
df['filtered_text'] = df['tokenize_text'].apply(lambda x: [word for word in x if word not in stop_words])
df['stem_text'] = df['filtered_text'].apply(lambda x: [stemmer.stem(word) for word in x])
df['lemma_text'] = df['filtered_text'].apply(lambda x: [lemma.lemmatize(word) for word in x])


# %%
x = df['stem_text'].apply(lambda x: ' '.join(x))
y = df['Sentiment']  

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

tfidf1 = TfidfVectorizer()
X_train = tfidf1.fit_transform(X_train)
X_test = tfidf1.transform(X_test)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

num_classes = len(np.unique(y_train))  
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

Model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')  
])
Model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
Model.fit(X_train, y_train, epochs=1, batch_size=32)  
joblib.dump(Model,'Model.pkl')
joblib.dump(tfidf1, 'tfidf1.pkl')



# %%

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import streamlit as st

Model = joblib.load('Model.pkl')
tfidf1_vectorizer = joblib.load('tfidf1.pkl')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

class_indices_to_sentiments = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

def predict_sentiment(review):
    cleaned_review = re.sub('<.*?>', "", review)
    cleaned_review = re.sub(r'[^\w\s]', '', cleaned_review)
    cleaned_review = cleaned_review.lower()
    tokenized_review = word_tokenize(cleaned_review)
    filter_review = [word for word in tokenized_review if word not in stop_words]
    stemmed_review = [stemmer.stem(word) for word in filter_review]
    tfidf_review = tfidf1_vectorizer.transform([' '.join(stemmed_review)])
    
    if tfidf_review.nnz == 0:
        return 'Unable to predict'

    sentiment_prediction = Model.predict(tfidf_review)
    predicted_class_index = np.argmax(sentiment_prediction, axis=1)

    return class_indices_to_sentiments.get(predicted_class_index[0], 'Unable to predict')

st.title('Hotel Review Sentiment Prediction:')
review_to_predict = st.text_area('Enter your review:')
if st.button('Predict Sentiment:'):
    predicted_sentiment = predict_sentiment(review_to_predict)
    st.write('Predicted sentiment:', predicted_sentiment)


# %%
#!ipynb-py-convert hotel_reviewAnalysis.ipynb hotelReviewAnalysis.py