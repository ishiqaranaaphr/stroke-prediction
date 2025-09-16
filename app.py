# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title='Stroke Prediction', layout='centered')

st.title('💙 Stroke Prediction - Logistic Regression')
st.markdown('Aplikasi sederhana untuk memprediksi kemungkinan stroke menggunakan model Logistic Regression.')

# Load artifacts
@st.cache_data
def load_artifacts():
    model = joblib.load('model_artifacts/model.joblib')
    scaler = joblib.load('model_artifacts/scaler.joblib')
    with open('model_artifacts/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, scaler, encoders

model, scaler, encoders = load_artifacts()

# Sidebar input
st.sidebar.header('Input Data Pasien')

def user_input_form():
    gender = st.sidebar.selectbox('Gender', options=['Male','Female','Other'])
    age = st.sidebar.slider('Age', 0, 120, 65)
    hypertension = st.sidebar.selectbox('Hypertension', options=[0,1])
    heart_disease = st.sidebar.selectbox('Heart disease', options=[0,1])
    ever_married = st.sidebar.selectbox('Ever married', options=['Yes','No'])
    work_type = st.sidebar.selectbox('Work type', options=['Private','Self-employed','Govt_job','children','Never_worked'])
    Residence_type = st.sidebar.selectbox('Residence type', options=['Urban','Rural'])
    avg_glucose_level = st.sidebar.number_input('Average glucose level', min_value=0.0, max_value=500.0, value=100.0)
    bmi = st.sidebar.number_input('BMI', min_value=5.0, max_value=70.0, value=25.0)
    smoking_status = st.sidebar.selectbox('Smoking status', options=['never smoked','formerly smoked','smokes','Unknown'])

    data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': Residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }
    return pd.DataFrame([data])

input_df = user_input_form()

# Encode categorical input safely
def safe_transform(input_df, encoders):
    df = input_df.copy()
    for col, le in encoders.items():
        if col in df.columns:
            val = df.at[0, col]
            try:
                df[col] = le.transform([str(val)])[0]
            except Exception:
                # jika nilai baru, fallback ke nilai paling umum (mode) yang ada pada encoder
                # ambil kelas pertama sebagai fallback
                df[col] = 0
    return df

input_encoded = safe_transform(input_df, encoders)

# Pastikan urutan kolom seperti saat training
feature_order = list(pd.read_csv('model_artifacts/sample_features.csv').columns)
input_encoded = input_encoded[feature_order]

# Scale
input_scaled = scaler.transform(input_encoded)

# Predict
pred = model.predict(input_scaled)[0]
pred_proba = model.predict_proba(input_scaled)[0,1]

# Hasil
st.subheader('Hasil Prediksi')
if st.button('Predict'):
    if pred == 1:
        st.error(f'🚨 Terindikasi kemungkinan stroke (Probabilitas = {pred_proba:.4f})')
    else:
        st.success(f'✅ Tidak terindikasi stroke (Probabilitas = {pred_proba:.4f})')

# Show input details and probability
with st.expander('Input Data (encoded)'):
    st.write(input_encoded)

# Feature importance (approx) — koefisien LR
with st.expander('Model Insight'):
    coef = model.coef_[0]
    feats = feature_order
    imp_df = pd.DataFrame({'feature': feats, 'coefficient': coef})
    imp_df = imp_df.reindex(imp_df.coefficient.abs().sort_values(ascending=False).index)
    st.table(imp_df.head(10))

st.markdown('---')
st.caption('Catatan: Aplikasi ini hanya demonstrasi akademik. Untuk penggunaan klinis diperlukan validasi lebih lanjut.')