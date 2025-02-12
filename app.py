import streamlit as st
import pandas as pd
import joblib

# Load model
dt_model = joblib.load("decision_tree_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")

# Judul aplikasi
st.title("Prediksi Hasil Belajar Siswa")

# Input data pengguna
gender = st.selectbox("Gender", ["Laki-laki", "Perempuan"])
absen = st.number_input("Jumlah Absen", min_value=0, max_value=50, value=5)
n_sikap = st.selectbox("Nilai Sikap", ["A", "B", "C"])
ekstra = st.selectbox("Ekstrakurikuler", ["Ya", "Tidak"])

# Konversi input menjadi dataframe
data = pd.DataFrame([[gender, absen, n_sikap, ekstra]], 
                    columns=["gender", "absen", "n_sikap", "ekstra"])

# Tombol Prediksi
if st.button("Prediksi"):
    pred_dt = dt_model.predict(data)[0]
    pred_rf = rf_model.predict(data)[0]
    st.write(f"**Prediksi Decision Tree:** {pred_dt}")
    st.write(f"**Prediksi Random Forest:** {pred_rf}")
