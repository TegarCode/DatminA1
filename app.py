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
                if st.button("Prediksi Sentimen"):
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

                    # Evaluasi Akurasi jika ada label 'sentiment2'
                    if 'sentiment2' in data.columns:
                        # Menghitung akurasi
                        accuracy = accuracy_score(data['sentiment2'], predictions)
                        st.success(f"Akurasi Model: {accuracy:.2%}")
                        st.write("Laporan Klasifikasi:")
                        st.text(classification_report(data['sentiment2'], predictions))
                    else:
                        st.warning("Kolom 'sentiment2' tidak ditemukan. Tidak dapat menghitung akurasi.")

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
