import streamlit as st
import joblib

# Load trained model and TF-IDF vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# App Title
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")
st.markdown("Enter a news article text or headline to check if it's **Fake** or **Real**.")

# Text Input
user_input = st.text_area("Enter news text here", height=200)

# Predict Button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        # Preprocess and predict
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        
        # Display Result
        if prediction == 1:
            st.success("✅ This news is **REAL**.")
        else:
            st.error("🚨 This news is **FAKE**.")
