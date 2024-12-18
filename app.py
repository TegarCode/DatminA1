import streamlit as st
import pandas as pd
import pickle
import requests
from io import BytesIO
from sklearn.metrics import accuracy_score
import plotly.express as px
import re


# Fungsi untuk mengunduh file dan memuat dengan pickle
def load_model_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pickle.load(BytesIO(response.content))
    else:
        st.error(f"Gagal mengunduh file dari URL: {url}")
        return None



#  Fungsi preprocessing teks
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Hilangkan karakter khusus (jika perlu)
    text = ''.join(e for e in text if e.isalnum() or e.isspace())
    return text.strip()

# Fungsi utama untuk aplikasi
def main():
    # Title untuk aplikasi
    st.title("Prediksi dan Analisis Sentimen 2024")

    # Bagian untuk upload file
    uploaded_file = st.file_uploader("Upload file CSV Anda", type=["csv"])
    if uploaded_file is not None:
        # Load data
        data = pd.read_csv(uploaded_file)
        st.write("Data yang diunggah:")
        st.write(data)

        # Load model dan vectorizer dari URL
        model_url = "https://raw.githubusercontent.com/TegarCode/DatminA1/main/svm_sentiment_model.pkl"
        vectorizer_url = "https://raw.githubusercontent.com/TegarCode/DatminA1/main/vectorizer.pkl"

        model = load_model_from_url(model_url)
        vectorizer = load_model_from_url(vectorizer_url)

        # Pastikan model dan vectorizer berhasil di-load
        if model and vectorizer:
            # Validasi kolom 'full_text'
            if 'full_text' in data.columns:
                # Transformasi data menggunakan vectorizer
                X_test = vectorizer.transform(data['full_text'])

                # Prediksi Sentimen
                if st.button("Prediksi Sentimen", key="prediksi_file"):
                    # Prediksi dengan model yang sudah dilatih
                    predictions = model.predict(X_test)

                    # Tambahkan hasil prediksi ke data
                    data['Predicted Sentiment'] = predictions

                    # Tampilkan hasil prediksi
                    st.write("Hasil Prediksi Sentimen:")
                    st.write(data[['full_text', 'Predicted Sentiment']])

                    # Visualisasi distribusi sentimen
                    sentiment_counts = data['Predicted Sentiment'].value_counts()
                    fig_bar = px.bar(
                        sentiment_counts,
                        x=sentiment_counts.index,
                        y=sentiment_counts.values,
                        labels={'x': 'Sentimen', 'y': 'Jumlah'},
                        title="Distribusi Sentimen"
                    )
                    st.plotly_chart(fig_bar)

                    # Tombol untuk mengunduh hasil
                    st.download_button(
                        label="Download Hasil Prediksi",
                        data=data.to_csv(index=False),
                        file_name="hasil_prediksi.csv",
                        mime="text/csv"
                    )
            else:
                st.error("Kolom 'full_text' tidak ditemukan dalam file yang diunggah.")

    # Title untuk aplikasi
    st.title("Prediksi dan Analisis Sentimen Tweet")

      # Bagian untuk input tweet manual
    st.title("Prediksi Sentimen Tweet Manual")
    tweet_input = st.text_area("Masukkan tweet di sini:")

    if tweet_input:
        # Load model dan vectorizer
        model_url = "https://raw.githubusercontent.com/TegarCode/DatminA1/main/svm_sentiment_model.pkl"
        vectorizer_url = "https://raw.githubusercontent.com/TegarCode/DatminA1/main/vectorizer.pkl"

        model = load_model_from_url(model_url)
        vectorizer = load_model_from_url(vectorizer_url)

        # Validasi model dan vectorizer
        if model and vectorizer:
            # Preprocessing input teks manual
            tweet_input_processed = preprocess_text(tweet_input)
            tweet_vect = vectorizer.transform([tweet_input_processed])

            # Prediksi Sentimen
            if st.button("Prediksi Sentimen", key="predict_manual"):
                sentiment = model.predict(tweet_vect)[0]

                # Pastikan hanya positif/negatif
                if sentiment == 'netral':
                    sentiment = 'negatif'  # Default jika netral muncul

                # Tampilkan hasil prediksi
                if sentiment == 'positif':
                    st.success(f"Sentimen tweet ini adalah: **Positif**")
                else:
                    st.error(f"Sentimen tweet ini adalah: **Negatif**")

if __name__ == '__main__':
    main()
