# Versi final aplikasi
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

# Import semua yang dibutuhkan untuk training
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import lightgbm as lgb

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Harga Sewa Properti",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- FUNGSI-FUNGSI UTAMA ---

@st.cache_data
def load_data(file_path):
    """Memuat data mentah dari CSV."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"File data '{file_path}' tidak ditemukan.")
        return None

# Fungsi ini sekarang akan melatih model JIKA BELUM ADA
@st.cache_resource
def train_and_get_model(model_path, data_path):
    """
    Memuat model jika sudah ada. Jika tidak,
    latih model dari awal dan simpan.
    """
    if not os.path.exists(model_path):
        with st.spinner(f"Model tidak ditemukan. Memulai proses pelatihan sekali jalan... (Ini mungkin butuh beberapa menit)"):
            # 1. Muat dan bersihkan data
            df_final = load_data(data_path)
            if df_final is None:
                return None

            # Pembersihan data dasar
            df_final['Harga Sewa'] = pd.to_numeric(df_final['Harga Sewa'], errors='coerce')
            df_final.dropna(subset=['Harga Sewa'], inplace=True)
            quantile_99 = df_final['Harga Sewa'].quantile(0.99)
            df_final = df_final[df_final['Harga Sewa'] < quantile_99].copy()
            df_final['Harga Sewa Log'] = np.log1p(df_final['Harga Sewa'])
            
            # --- PERBAIKAN: Buang kolom yang tidak ada di input form ---
            # Ini akan memastikan model dilatih pada fitur yang sama dengan yang diinput pengguna
            if 'Kamar Tidur Pembantu' in df_final.columns:
                df_final.drop(columns=['Kamar Tidur Pembantu', 'Kamar Mandi Pembantu'], inplace=True)
            # -----------------------------------------------------------

            # 2. Pisahkan Fitur (X) dan Target (y)
            X = df_final.drop(columns=['Harga Sewa', 'Harga Sewa Log'])
            y = df_final['Harga Sewa Log']

            # 3. Buat Pipeline Preprocessing
            numeric_features = X.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X.select_dtypes(include=['object']).columns.tolist()

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])

            # 4. Buat dan Latih Pipeline Final
            final_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', lgb.LGBMRegressor(random_state=70))
            ])

            final_pipeline.fit(X, y)
            
            # 5. Simpan Pipeline
            joblib.dump(final_pipeline, model_path)
            st.success("Pelatihan selesai! Model berhasil dibuat dan disimpan.")
            return final_pipeline
    else:
        # Jika file sudah ada, langsung muat
        model = joblib.load(model_path)
        return model

# --- UI Aplikasi ---

st.title("🏠 Prediksi Harga Rumah (LightGBM)")
st.markdown("Aplikasi ini menggunakan model LightGBM untuk estimasi harga sewa.")

DATA_PATH = 'data_fix.csv'
MODEL_PATH = 'model_prediksi_final_fix.joblib'

# Panggil fungsi utama untuk mendapatkan model (melatih atau memuat)
model_pipeline = train_and_get_model(MODEL_PATH, DATA_PATH)
raw_df = load_data(DATA_PATH)

# --- Sidebar untuk Input ---
with st.sidebar:
    st.header("📝 Masukkan Detail Properti")
    
    if raw_df is None:
        st.error(f"File data '{DATA_PATH}' tidak ditemukan.")
    else:
        with st.form(key='prediction_form'):
            raw_df['Kota'] = raw_df['Kota'].apply(lambda x: 'Jakarta' if isinstance(x, str) and 'Jakarta' in x else x)
            
            kota_options = sorted(raw_df['Kota'].dropna().unique())
            sertifikat_options = sorted(raw_df['Sertifikat'].dropna().unique())
            kondisi_options = sorted(raw_df['Kondisi Properti'].dropna().unique())

            kota = st.selectbox("Kota", options=kota_options)
            kamar_tidur = st.slider("Kamar Tidur", 1, 10, 3)
            kamar_mandi = st.slider("Kamar Mandi", 1, 10, 2)
            luas_tanah = st.number_input("Luas Tanah (m²)", min_value=30, value=120)
            luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=30, value=100)
            jumlah_lantai = st.slider("Jumlah Lantai", 1, 5, 1)
            carport = st.slider("Carport (mobil)", 0, 10, 1)
            garasi = st.slider("Garasi (mobil)", 0, 10, 0)
            daya_listrik = st.number_input("Daya Listrik (VA)", min_value=900, value=2200, step=100)
            sertifikat = st.selectbox("Sertifikat", options=sertifikat_options)
            kondisi_properti = st.selectbox("Kondisi Properti", options=kondisi_options)

            submit_button = st.form_submit_button(label='✨ Prediksi Harga')

# --- Tampilan Utama ---
if model_pipeline is None:
    st.error(f"Gagal melatih atau memuat model. Periksa file data dan log.")
elif submit_button:
    with st.spinner('Model sedang menganalisis...'):
        time.sleep(1)
        
        input_data = pd.DataFrame([{
            'Kamar Tidur': kamar_tidur, 'Kamar Mandi': kamar_mandi,
            'Luas Tanah': luas_tanah, 'Luas Bangunan': luas_bangunan,
            'Jumlah Lantai': jumlah_lantai, 'Carport': carport,
            'Garasi': garasi, 'Daya Listrik': daya_listrik,
            'Sertifikat': sertifikat, 'Kondisi Properti': kondisi_properti,
            'Kota': kota
        }])

        prediction_log = model_pipeline.predict(input_data)
        prediction_final = np.expm1(prediction_log[0])

        st.subheader("Estimasi Harga Sewa Tahunan")
        st.metric(label="Harga Prediksi", value=f"Rp {prediction_final:,.0f}")
        st.info("Prediksi ini dibuat menggunakan model LightGBM.", icon="💡")
else:
    st.info("Silakan isi formulir di sidebar dan klik tombol prediksi untuk melihat hasilnya.")




































# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import time

# # --- Konfigurasi Halaman ---
# st.set_page_config(
#     page_title="Prediksi Harga (LightGBM)",
#     page_icon="🏠",
#     layout="centered", # Menggunakan layout centered untuk stabilitas
#     initial_sidebar_state="expanded"
# )

# # --- Fungsi Bantuan ---
# @st.cache_data
# def load_data(file_path):
#     """Memuat data mentah untuk opsi UI."""
#     try:
#         df = pd.read_csv(file_path)
#         return df
#     except FileNotFoundError:
#         return None

# @st.cache_resource
# def load_model(model_path):
#     """Memuat pipeline model yang sudah dilatih."""
#     try:
#         model = joblib.load(model_path)
#         return model
#     except Exception as e:
#         st.error(f"Gagal memuat model: {e}")
#         return None

# # --- UI Aplikasi ---

# st.title("🏠 House Pricing Prediction App")
# st.markdown("Source by Rumah123.com | Proyek Machine Learning")

# # Path ke file
# DATA_PATH = 'database_sewa_rumah_fix - database_sewa_rumahfix.csv.csv'
# MODEL_PATH = 'model_prediksi_final.joblib' # <-- MENGGUNAKAN MODEL LIGHTGBM

# # Muat semua aset
# raw_df = load_data(DATA_PATH)
# model_pipeline = load_model(MODEL_PATH)

# # --- Sidebar untuk Input ---
# with st.sidebar:
#     st.header("📝 Masukkan Detail Properti")
    
#     if raw_df is None:
#         st.error(f"File data '{DATA_PATH}' tidak ditemukan.")
#     else:
#         with st.form(key='prediction_form'):
#             # Gabungkan area Jakarta di opsi dropdown
#             raw_df['Kota'] = raw_df['Kota'].apply(lambda x: 'Jakarta' if isinstance(x, str) and 'Jakarta' in x else x)
            
#             kota_options = sorted(raw_df['Kota'].dropna().unique())
#             sertifikat_options = sorted(raw_df['Sertifikat'].dropna().unique())
#             kondisi_options = sorted(raw_df['Kondisi Properti'].dropna().unique())

#             kota = st.selectbox("Kota", options=kota_options)
#             kamar_tidur = st.slider("Kamar Tidur", 1, 10, 3)
#             kamar_mandi = st.slider("Kamar Mandi", 1, 10, 2)
#             luas_tanah = st.number_input("Luas Tanah (m²)", min_value=30, value=120)
#             luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=30, value=100)
#             jumlah_lantai = st.slider("Jumlah Lantai", 1, 5, 1)
#             carport = st.slider("Carport (mobil)", 0, 10, 1)
#             garasi = st.slider("Garasi (mobil)", 0, 10, 0)
#             daya_listrik = st.number_input("Daya Listrik (VA)", min_value=900, value=2200, step=100)
#             sertifikat = st.selectbox("Sertifikat", options=sertifikat_options)
#             kondisi_properti = st.selectbox("Kondisi Properti", options=kondisi_options)

#             submit_button = st.form_submit_button(label='✨ Prediksi Harga')

# # --- Tampilan Utama ---
# if model_pipeline is None:
#     st.error(f"File model '{MODEL_PATH}' tidak ditemukan. Pastikan file ini ada di repositori Anda.")
# elif submit_button:
#     with st.spinner('Model sedang menganalisis...'):
#         time.sleep(1)
        
#         input_data = pd.DataFrame([{
#             'Kamar Tidur': kamar_tidur, 'Kamar Mandi': kamar_mandi,
#             'Luas Tanah': luas_tanah, 'Luas Bangunan': luas_bangunan,
#             'Jumlah Lantai': jumlah_lantai, 'Carport': carport,
#             'Garasi': garasi, 'Daya Listrik': daya_listrik,
#             'Sertifikat': sertifikat, 'Kondisi Properti': kondisi_properti,
#             'Kota': kota
#         }])

#         # Gunakan pipeline untuk memprediksi
#         prediction_log = model_pipeline.predict(input_data)

#         # Kembalikan ke skala Rupiah
#         prediction_final = np.expm1(prediction_log[0])

#         st.subheader("Estimasi Harga Sewa Tahunan")
#         st.metric(label="Harga Prediksi", value=f"Rp {prediction_final:,.0f}")
#         st.info("Prediksi ini dibuat menggunakan model LightGBM.", icon="💡")
# else:
#     st.info("Silakan isi formulir di sidebar dan klik tombol prediksi untuk melihat hasilnya.")

















# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.linear_model import Ridge
# from sklearn.metrics import r2_score
# import time

# # Konfigurasi halaman Streamlit
# st.set_page_config(
#     page_title="Prediksi Harga Sewa Rumah | Proyek ML",
#     page_icon="🏠",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --- FUNGSI-FUNGSI UTAMA ---

# @st.cache_data
# def load_data(file_path):
#     try:
#         df = pd.read_csv(file_path)
#         return df
#     except FileNotFoundError:
#         st.error(f"File tidak ditemukan di path: {file_path}")
#         return None

# @st.cache_data
# def preprocess_data(_df):
#     df = _df.copy()

#     # 1. Rename columns
#     df.rename(columns={
#         'Harga Sewa': 'harga_sewa', 'Kamar Tidur': 'kamar_tidur', 'Kamar Mandi': 'kamar_mandi',
#         'Luas Tanah': 'luas_tanah', 'Luas Bangunan': 'luas_bangunan', 'Jumlah Lantai': 'jumlah_lantai',
#         'Carport': 'carport', 'Garasi': 'garasi', 'Sertifikat': 'sertifikat',
#         'Daya Listrik': 'daya_listrik', 'Kondisi Properti': 'kondisi_properti',
#         'Kamar Tidur Pembantu': 'kt_pembantu', 'Kamar Mandi Pembantu': 'km_pembantu', 'Kota': 'kota'
#     }, inplace=True)

#     # 2. Clean and convert 'harga_sewa' to numeric
#     df['harga_sewa'] = pd.to_numeric(df['harga_sewa'], errors='coerce')
#     df.dropna(subset=['harga_sewa'], inplace=True)
#     df = df[df['harga_sewa'] > 0]

#     # 3. Impute missing values for primary features
#     for col in ['kamar_tidur', 'kamar_mandi', 'luas_bangunan', 'jumlah_lantai']:
#         df[col].fillna(df[col].median(), inplace=True)

#     for col in ['kondisi_properti', 'sertifikat', 'daya_listrik']:
#         if df[col].isnull().any():
#             df[col].fillna(df[col].mode()[0], inplace=True)

#     # 4. Clean 'Lainnya' values
#     for kolom in ['sertifikat', 'kondisi_properti', 'daya_listrik']:
#         if 'Lainnya' in df[kolom].unique():
#             try:
#                 modus_value = df[df[kolom] != 'Lainnya'][kolom].mode()[0]
#                 df[kolom] = df[kolom].replace('Lainnya', modus_value)
#             except IndexError:
#                 pass

#     # 5. Convert 'daya_listrik' to numeric
#     df['daya_listrik'] = pd.to_numeric(df['daya_listrik'], errors='coerce')
#     df['daya_listrik'].fillna(df['daya_listrik'].median(), inplace=True)
    
#     # 6. Drop irrelevant columns
#     df.drop(['kt_pembantu', 'km_pembantu'], axis=1, inplace=True)

#     # 7. Feature Engineering
#     df['harga_sewa_log'] = np.log1p(df['harga_sewa'])
    
#     X = df.drop(['harga_sewa', 'harga_sewa_log'], axis=1)
#     y = df['harga_sewa_log']

#     # 8. One-Hot Encoding
#     X = pd.get_dummies(X, columns=['kota', 'sertifikat', 'kondisi_properti'], drop_first=True)

#     # =================================================================
#     # --- Sapu Bersih Semua NaN yang Tersisa di X ---
#     # =================================================================
#     # Loop melalui semua kolom di X dan isi NaN yang mungkin masih ada
#     # dengan median dari masing-masing kolom.
#     for col in X.columns:
#         if X[col].isnull().any():
#             X[col].fillna(X[col].median(), inplace=True)
#     # =================================================================

#     return X, y

# @st.cache_resource
# def train_and_evaluate_models(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     models = {
#         "Ridge": Ridge(random_state=42),
#         "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
#         "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=100)
#     }
#     scores = {}
#     for name, model in models.items():
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         scores[name] = r2_score(y_test, y_pred)

#     best_model_name = max(scores, key=scores.get)
#     best_model = models[best_model_name]
#     best_score = scores[best_model_name]
    
#     best_model.fit(X, y) # Train final model on all data
    
#     return best_model, best_model_name, best_score, scores

# # --- STREAMLIT UI ---

# st.title("🏠 House Pricing Prediction App")
# st.markdown("Source by Rumah123.com | Proyek Machine Learning")

# data_path = 'database_sewa_rumah_fix - database_sewa_rumahfix.csv.csv'
# raw_df = load_data(data_path)

# if raw_df is not None:
#     X, y = preprocess_data(raw_df.copy())
#     model, model_name, score, all_scores = train_and_evaluate_models(X, y)

#     st.sidebar.header("Masukkan Fitur Properti Anda")
#     with st.sidebar.form(key='prediction_form'):
#         temp_df = raw_df.copy()
        
#         # Clean options for selectbox
#         for col in ['Kota', 'Sertifikat', 'Kondisi Properti']:
#             if 'Lainnya' in temp_df[col].unique():
#                 temp_df[col] = temp_df[col].replace('Lainnya', temp_df[col].mode()[0])

#         kota_options = sorted(temp_df['Kota'].dropna().unique())
#         sertifikat_options = sorted(temp_df['Sertifikat'].dropna().unique())
#         kondisi_options = sorted(temp_df['Kondisi Properti'].dropna().unique())

#         kota = st.selectbox("Kota", options=kota_options)
#         kamar_tidur = st.slider("Jumlah Kamar Tidur", 1, 10, 3)
#         kamar_mandi = st.slider("Jumlah Kamar Mandi", 1, 10, 2)
#         luas_tanah = st.number_input("Luas Tanah (m²)", min_value=30, value=120)
#         luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=30, value=100)
#         jumlah_lantai = st.selectbox("Jumlah Lantai", [1, 2, 3, 4, 5])
#         carport = st.slider("Carport (mobil)", 0, 10, 1)
#         garasi = st.slider("Garasi (mobil)", 0, 10, 0)
#         daya_listrik = st.number_input("Daya Listrik (VA)", min_value=900, value=2200)
#         sertifikat = st.selectbox("Jenis Sertifikat", options=sertifikat_options)
#         kondisi = st.selectbox("Kondisi Properti", options=kondisi_options)
        
#         submit_button = st.form_submit_button(label='✨ Prediksi Harga!')

#     # Main page layout
#     col1, col2 = st.columns((2, 1.5))
#     with col1:
#         st.subheader("Analisis Model & Data")
#         st.markdown(f"Model terbaik: **{model_name}** (Skor R²: **{score:.2f}**)")
#         st.write("Perbandingan Skor Antar Model:")
#         st.bar_chart(pd.Series(all_scores, name="Skor R²"))
#         with st.expander("Lihat Data Siap Latih"):
#             st.dataframe(X.head())

#     with col2:
#         st.subheader("Hasil Prediksi Anda")
#         if submit_button:
#             with st.spinner('Memprediksi...'):
#                 input_data = pd.DataFrame([{
#                     'kamar_tidur': kamar_tidur, 'kamar_mandi': kamar_mandi, 'luas_tanah': luas_tanah,
#                     'luas_bangunan': luas_bangunan, 'carport': carport, 'daya_listrik': daya_listrik,
#                     'garasi': garasi, 'jumlah_lantai': jumlah_lantai, 'kota': kota,
#                     'sertifikat': sertifikat, 'kondisi_properti': kondisi
#                 }])
                
#                 input_encoded = pd.get_dummies(input_data)
#                 final_input = input_encoded.reindex(columns=X.columns, fill_value=0)

#                 # Final check for NaNs in user input (safety net)
#                 final_input.fillna(0, inplace=True)
                
#                 prediction_log = model.predict(final_input)
#                 prediction_final = np.expm1(prediction_log)[0]
                
#                 st.success("Estimasi Harga Sewa Tahunan:")
#                 st.markdown(f"<h2 style='text-align: center; color: #28a745;'>Rp {prediction_final:,.0f}</h2>", unsafe_allow_html=True)
#         else:
#             st.info("Isi formulir di kiri untuk mendapatkan prediksi.")
# else:
#     st.error("Gagal memuat file data.")

    














