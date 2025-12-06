import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')

model = joblib.load('text_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
le = joblib.load('LabelBinarizer.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text).strip()
    text = re.sub(r'\s+', ' ', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

st.title("📝 Movie Review Classifier")
st.write("Enter your review below to get a prediction:")

user_input = st.text_area("Type here...", "")

if st.button("Predict"):
    if user_input.strip() != "":
        cleaned_input = [clean_text(user_input)]
        
        input_tfidf = vectorizer.transform(cleaned_input)
        
        pred_numeric = model.predict(input_tfidf)
        pred_label = le.inverse_transform(pred_numeric)
        
        st.success(f"Prediction: **{pred_label[0]}**")
    else:
        st.warning("Please enter some text for prediction!")
