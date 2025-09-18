import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detector")
st.subheader("Enter a news article below:")

st.sidebar.title("📌 About")
st.sidebar.write("This app uses a machine learning model to detect fake news.")

# Input
user_input = st.text_area("✏️ News Article", height=200)

if st.button("🧠 Analyze"):
    if user_input.strip():
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        proba = model.predict_proba(transformed_input)[0]
        confidence = round(proba[prediction] * 100, 2)

        if prediction == 1:
            st.success(f"✅ Real News — Confidence: {confidence}%")
            st.progress(int(confidence))
        else:
            st.error(f"🚫 Fake News — Confidence: {confidence}%")
            st.progress(int(confidence))
    else:
        st.warning("⚠️ Please enter a news article to classify.")
