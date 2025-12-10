import joblib
import streamlit as st

# Load Model & Vectorizer
model = joblib.load("models/model_logistic_regression.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

st.title("Aplikasi Klasifikasi Komentar Publik")
st.write("Masukkan komentar anda pada kolom dibawah ini:")

# Input pengguna
user_input = st.text_input("Masukkan komentar anda:")

# Tombol prediksi
if st.button("Prediksi"):
    if user_input.strip() != "":
        st.warning("Sedang memproses...")

        # Transformasi dan prediksi
        vector = tfidf.transform([user_input])
        prediksi = model.predict(vector)[0]

        # Mapping label
        label_map = {0: "Negatif", 1: "Positif"}

        st.subheader("Hasil Prediksi:")
        st.write(f"Komentar Anda bernada: **{label_map.get(prediksi, prediksi)}**")
    else:
        st.error("Input tidak boleh kosong!")
