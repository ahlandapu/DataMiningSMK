import streamlit as st
import pandas as pd
import joblib

# Load model
rf_model = joblib.load("random_forest_model.pkl")

# Judul aplikasi
st.title("Prediksi Hasil Belajar Siswa")

# Input data pengguna sesuai dengan fitur saat pelatihan
gender = st.selectbox("Gender", ["Laki-laki", "Perempuan"])
absen = st.number_input("Jumlah Absen", min_value=0, max_value=50, value=5)
n_sikap = st.selectbox("Nilai Sikap", ["A", "B", "C"])
organisasi = st.selectbox("Organisasi", ["Ya", "Tidak"])
ekstra = st.selectbox("Ekstrakurikuler", ["Ya", "Tidak"])
asal_smp = st.selectbox("Asal SMP", ["Negeri", "Swasta"])
tmpt_tinggal = st.selectbox("Tempat Tinggal", ["Kota", "Desa"])
status_nikah = st.selectbox("Status Pernikahan Orang Tua", ["Menikah", "Cerai", "Wafat"])
pendidikan_ayah = st.selectbox("Pendidikan Ayah", ["SD", "SMP", "SMA", "S1", "S2", "S3"])
pekerjaan_ayah = st.selectbox("Pekerjaan Ayah", ["PNS", "Swasta", "Wirausaha", "Tidak Bekerja"])
gaji_ayah = st.number_input("Gaji Ayah", min_value=0, value=1000000)
pendidikan_ibu = st.selectbox("Pendidikan Ibu", ["SD", "SMP", "SMA", "S1", "S2", "S3"])
pekerjaan_ibu = st.selectbox("Pekerjaan Ibu", ["PNS", "Swasta", "Wirausaha", "Tidak Bekerja"])
gaji_ibu = st.number_input("Gaji Ibu", min_value=0, value=1000000)
beasiswa = st.selectbox("Beasiswa", ["Ya", "Tidak"])
jml_keluarga = st.number_input("Jumlah Keluarga", min_value=1, max_value=10, value=4)
status_rumah = st.selectbox("Status Rumah", ["Milik Sendiri", "Sewa", "Menumpang"])

# Konversi input menjadi DataFrame
data = pd.DataFrame([[gender, absen, n_sikap, organisasi, ekstra, asal_smp, tmpt_tinggal, status_nikah, 
                      pendidikan_ayah, pekerjaan_ayah, gaji_ayah, pendidikan_ibu, pekerjaan_ibu, gaji_ibu, 
                      beasiswa, jml_keluarga, status_rumah]], 
                    columns=['gender', 'absen', 'n_sikap', 'organisasi', 'ekstra', 'asal_smp', 'tmpt_tinggal', 
                             'status_nikah', 'pendidikan_ayah', 'pekerjaan_ayah', 'gaji_ayah', 'pendidikan_ibu', 
                             'pekerjaan_ibu', 'gaji_ibu', 'beasiswa', 'jml_keluarga', 'status_rumah'])

# Pastikan fitur sesuai dengan model yang telah dilatih
data = data.reindex(columns=['gender', 'absen', 'n_sikap', 'organisasi', 'ekstra', 'asal_smp', 'tmpt_tinggal', 
                             'status_nikah', 'pendidikan_ayah', 'pekerjaan_ayah', 'gaji_ayah', 'pendidikan_ibu', 
                             'pekerjaan_ibu', 'gaji_ibu', 'beasiswa', 'jml_keluarga', 'status_rumah'], fill_value=0)

# Tombol Prediksi
if st.button("Prediksi"):
    pred_rf = rf_model.predict(data)[0]
    st.write(f"**Prediksi Random Forest:** {pred_rf}")
