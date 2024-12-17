import streamlit as st
import pandas as pd
import pickle
import requests
from io import BytesIO
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px

# Fungsi untuk mengunduh file dan memuat dengan pickle
def load_model_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pickle.load(BytesIO(response.content))
    else:
        st.error(f"Gagal mengunduh file dari URL: {url}")
        return None

# Fungsi utama untuk aplikasi
def main():
    # Title untuk aplikasi
    st.title("Prediksi dan Analisis Sentimen Tweet")

    # Bagian untuk input tweet
    tweet_input = st.text_area("Masukkan tweet di sini:")

    if tweet_input:
        # Load model dan vectorizer dari URL
        model_url = "https://raw.githubusercontent.com/TegarCode/DatminA1/main/svm_sentiment_model.pkl"
        vectorizer_url = "https://raw.githubusercontent.com/TegarCode/DatminA1/main/vectorizer.pkl"

        model = load_model_from_url(model_url)
        vectorizer = load_model_from_url(vectorizer_url)

        # Pastikan model dan vectorizer berhasil di-load
        if model and vectorizer:
            # Transformasi data menggunakan vectorizer
            tweet_vect = vectorizer.transform([tweet_input])

            # Prediksi Sentimen
            if st.button("Prediksi Sentimen"):
                # Prediksi dengan model yang sudah dilatih
                sentiment = model.predict(tweet_vect)

                # Tampilkan hasil prediksi
                if sentiment == 'positif':
                    st.success(f"Sentimen tweet ini adalah: **Positif**")
                elif sentiment == 'negatif':
                    st.error(f"Sentimen tweet ini adalah: **Negatif**")
                else:
                    st.info(f"Sentimen tweet ini adalah: **Netral**")

if __name__ == '__main__':
    main()
