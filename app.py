import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Mengatur konfigurasi halaman utama aplikasi
st.set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon="",
    layout="wide"
)

# 2. Memuat model terbaik yang telah disimpan (.pkl)
@st.cache_resource
def load_model():
    return joblib.load('model_churn.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# 3. Header dan deskripsi aplikasi
st.title(" Aplikasi Prediksi Churn Pelanggan")
st.markdown("""
Aplikasi ini dikembangkan untuk memprediksi potensi *churn* pelanggan (apakah pelanggan akan berhenti menggunakan layanan atau tetap bertahan).
Proyek ini merupakan bagian dari **Ujian Akhir Semester (UAS) Bengkel Koding Data Science - Universitas Dian Nuswantoro**.
""")

st.info("Silakan masukkan data pelanggan pada panel sebelah kiri untuk melakukan prediksi.")

# 4. Membuat form input fitur pada Sidebar
st.sidebar.header("Form Input Fitur Pelanggan")

with st.sidebar.form(key='churn_form'):
    # Fitur Demografi
    st.subheader("Informasi Demografi")
    gender = st.selectbox("Jenis Kelamin (gender)", ["Male", "Female"])
    age = st.number_input("Usia (age)", min_value=0, max_value=100, value=35)
    country = st.selectbox("Negara Asal (country)", ["India", "Germany", "USA", "UK"])
    
    # Fitur Layanan & Transaksi
    st.subheader("Aktivitas Layanan & Transaksi")
    subscription_type = st.selectbox("Tipe Langganan", ["Monthly", "Annual"])
    is_premium_user = st.radio("Pengguna Premium? (is_premium_user)", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    total_visits = st.slider("Total Kunjungan (total_visits)", min_value=0, max_value=50, value=15)
    avg_session_time = st.number_input("Rata-rata Waktu Sesi / menit (avg_session_time)", min_value=0.0, value=8.0)
    total_spent = st.number_input("Total Pengeluaran (total_spent)", min_value=0.0, value=500.0)
    avg_order_value = st.number_input("Rata-rata Nilai Pesanan (avg_order_value)", min_value=0.0, value=60.0)
    
    # Fitur Kepuasan & Interaksi
    st.subheader("Interaksi & Keluhan")
    support_tickets = st.slider("Jumlah Tiket Bantuan (support_tickets)", min_value=0, max_value=10, value=2)
    refund_requested = st.radio("Pernah Mengajukan Refund?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    satisfaction_score = st.slider("Skor Kepuasan (satisfaction_score)", min_value=1.0, max_value=5.0, value=3.5, step=0.5)
    nps_score = st.slider("NPS Score", min_value=0, max_value=10, value=5)
    last_3_month_freq = st.slider("Frekuensi Pembelian 3 Bulan Terakhir", min_value=0, max_value=20, value=7)
    
    # Tombol submit form
    submit_button = st.form_submit_button(label="Prediksi Churn")

# 5. Memproses input data dan melakukan inferensi model
if submit_button:
    # Mengemas input menjadi DataFrame yang strukturnya sesuai dengan pipeline model
    input_data = pd.DataFrame([{
        'gender': gender,
        'age': age,
        'country': country,
        'subscription_type': subscription_type,
        'is_premium_user': is_premium_user,
        'total_visits': total_visits,
        'avg_session_time': avg_session_time,
        'total_spent': total_spent,
        'avg_order_value': avg_order_value,
        'support_tickets': support_tickets,
        'refund_requested': refund_requested,
        'satisfaction_score': satisfaction_score,
        'nps_score': nps_score,
        'last_3_month_purchase_freq': last_3_month_freq
    }])
    
    st.subheader("Data Pelanggan yang Dimasukkan:")
    st.dataframe(input_data)
    
    # Melakukan prediksi menggunakan model (.predict() dan .predict_proba())
    try:
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # 6. Menampilkan Hasil Prediksi dengan elemen visual pendukung
        st.subheader("Hasil Analisis Prediksi:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error(" HASIL PREDIKSI: PELANGGAN BERPOTENSI CHURN (BERHENTI)")
            else:
                st.success(" HASIL PREDIKSI: PELANGGAN BERPOTENSI TETAP BERLANGGANAN (LOYAL)")
                
        with col2:
            st.metric(label="Probabilitas Churn", value=f"{prediction_proba[1]*100:.2f}%")
            st.metric(label="Probabilitas Bertahan", value=f"{prediction_proba[0]*100:.2f}%")
            
        # Memberikan visualisasi bar chart sederhana untuk probabilitas
        proba_df = pd.DataFrame({
            'Status': ['Bertahan (0)', 'Churn (1)'],
            'Probabilitas (%)': [prediction_proba[0]*100, prediction_proba[1]*100]
        })
        st.bar_chart(data=proba_df, x='Status', y='Probabilitas (%)')
        
    except Exception as e:
        st.warning("""
        Catatan: Pastikan format urutan kolom dan objek pipeline pre-processing data (Imputer/Encoder) 
        sudah terbungkus secara utuh di dalam file model_churn.pkl.
        """)
        st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")
