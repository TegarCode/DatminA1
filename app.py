import streamlit as st
import pandas as pd
import pickle
import requests
from io import BytesIO
from sklearn.metrics import accuracy_score
import plotly.express as px


# Fungsi untuk mengunduh file dan memuat dengan pickle
def load_model_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pickle.load(BytesIO(response.content))
    else:
        st.error(f"Gagal mengunduh file dari URL: {url}")
        return None


# Fungsi preprocessing teks
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Hilangkan karakter khusus
    text = ''.join(e for e in text if e.isalnum() or e.isspace())
    return text.strip()


# Fungsi utama untuk aplikasi
def main():
    # Title untuk aplikasi
    st.title("Prediksi dan Analisis Sentimen 2024")

    # Load model dan vectorizer dari URL
    model_url = "https://raw.githubusercontent.com/TegarCode/DatminA1/main/svm_sentiment_model.pkl"
    vectorizer_url = "https://raw.githubusercontent.com/TegarCode/DatminA1/main/vectorizer.pkl"

    model = load_model_from_url(model_url)
    vectorizer = load_model_from_url(vectorizer_url)

    # Pastikan model dan vectorizer berhasil di-load
    if model and vectorizer:
        # --- Bagian 1: Prediksi dari File ---
        st.header("Prediksi Sentimen dari File CSV")
        uploaded_file = st.file_uploader("Upload file CSV Anda", type=["csv"])
        if uploaded_file is not None:
            # Load data
            data = pd.read_csv(uploaded_file)
            st.write("Data yang diunggah:")
            st.write(data)

            # Validasi kolom 'full_text'
            if 'full_text' in data.columns:
                # Transformasi data menggunakan vectorizer
                X_test = vectorizer.transform(data['full_text'])

                # Prediksi Sentimen
                if st.button("Prediksi Sentimen (File)", key="prediksi_file"):
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

        # --- Bagian 2: Prediksi dari Input Manual ---
        st.header("Prediksi Sentimen dari Input Manual")
        user_input = st.text_area("Masukkan teks di bawah ini:", "")
        if st.button("Prediksi Sentimen (Manual)", key="prediksi_manual"):
            if user_input.strip() != "":
                # Preprocessing teks
                preprocessed_text = preprocess_text(user_input)
                # Transformasi teks menggunakan vectorizer
                X_input = vectorizer.transform([preprocessed_text])
                # Prediksi sentimen
                predicted_sentiment = model.predict(X_input)[0]

                # Tampilkan hasil prediksi
                st.write(f"**Teks:** {user_input}")
                st.write(f"**Prediksi Sentimen:** {predicted_sentiment}")
            else:
                st.error("Silakan masukkan teks sebelum memprediksi.")

    else:
        st.error("Gagal memuat model atau vectorizer. Pastikan URL valid.")


if __name__ == '__main__':
    main()
