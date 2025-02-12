import streamlit as st
import pandas as pd
import joblib

# Load model
dt_model = joblib.load("decision_tree_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")

# Judul aplikasi
st.title("Prediksi Hasil Belajar Siswa")

# Input data pengguna sesuai dengan fitur saat pelatihan
gender = st.selectbox("Gender", ["Laki-laki", "Perempuan"])
absen = st.number_input("Jumlah Absen", min_value=0, max_value=50, value=5)
n_sikap = st.selectbox("Nilai Sikap", ["A", "B", "C"])
ekstra = st.selectbox("Ekstrakurikuler", ["Ya", "Tidak"])
asal_smp = st.selectbox("Asal SMP", ["Negeri", "Swasta"])
beasiswa = st.selectbox("Beasiswa", ["Ya", "Tidak"])
gaji_ayah = st.number_input("Gaji Ayah", min_value=0, value=1000000)
gaji_ibu = st.number_input("Gaji Ibu", min_value=0, value=1000000)
jml_keluarga = st.number_input("Jumlah Keluarga", min_value=1, max_value=10, value=4)

# Konversi input menjadi dataframe
#data = pd.DataFrame([[gender, absen, n_sikap, ekstra]], 
#                    columns=["gender", "absen", "n_sikap", "ekstra"])
# Konversi input menjadi dataframe
data = pd.DataFrame([[gender, absen, n_sikap, ekstra, asal_smp, beasiswa, gaji_ayah, gaji_ibu, jml_keluarga]], 
                    columns=["gender", "absen", "n_sikap", "ekstra", "asal_smp", "beasiswa", "gaji_ayah", "gaji_ibu", "jml_keluarga"])

# Pastikan fitur sesuai dengan model yang telah dilatih
data = data.reindex(columns=["gender", "absen", "n_sikap", "ekstra", "asal_smp", "beasiswa", "gaji_ayah", "gaji_ibu", "jml_keluarga"], fill_value=0)

# Tombol Prediksi
if st.button("Prediksi"):
    pred_dt = dt_model.predict(data)[0]
    pred_rf = rf_model.predict(data)[0]
    st.write(f"**Prediksi Decision Tree:** {pred_dt}")
    st.write(f"**Prediksi Random Forest:** {pred_rf}")
