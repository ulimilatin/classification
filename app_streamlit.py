import streamlit as st
import pandas as pd
import joblib

# Load model
loaded_model = joblib.load("mlbb_model.joblib")

# Judul aplikasi
st.title("Machine Learning Classification")
st.markdown("Ini adalah aplikasi untuk memprediksi jenis role penyerang atau bertahan di MLBB")

# Input dari user
kill = st.slider("Jumlah Kill", 0, 20)
assist = st.slider("Jumlah Assist", 0, 20)
death = st.slider("Jumlah Death", 0, 20)
turret = st.slider("Jumlah Turret", 0, 20)

# Tombol prediksi
if st.button("Prediksi"):
    data_baru = pd.DataFrame(
        [[kill, assist, death, turret]],
        columns=["kill", "assist", "death", "turret"]
    )
    hasil = loaded_model.predict(data_baru)[0]
    st.success(f"Hasil Prediksi : {hasil}")
    st.balloons()

# Footer
st.caption("Dibuat dengan :heart: oleh Adi Setiawan")
