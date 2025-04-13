import streamlit as st
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load NLTK stopwords (make sure you have run nltk.download('stopwords') at least once)
# nltk.download('stopwords')
ps = PorterStemmer()

# Load your trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    return ' '.join(review)

def predict_sentiment(text):
    cleaned = preprocess_text(text)
    vectorized = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vectorized)
    return "😊 Positive" if prediction[0] == 1 else "😞 Negative"

# Streamlit UI
st.title("🎬 Movie Review Sentiment Analyzer")

user_input = st.text_area("Type your movie review here:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please type something first!")
    else:
        result = predict_sentiment(user_input)
        st.success(f"Prediction: {result}")