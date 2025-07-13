import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib # Library untuk menyimpan model dan scaler

# --- 1. Data (Sama seperti sebelumnya) ---
# Dataset simulasi Pima Indians Diabetes
data = {
    'Pregnancies': [6, 1, 8, 1, 0, 5, 3, 10, 2, 8, 5, 0, 1, 4, 1, 0, 5, 7, 10, 1],
    'Glucose': [148, 85, 183, 89, 137, 116, 78, 115, 197, 125, 135, 128, 79, 110, 160, 102, 129, 119, 168, 118],
    'BloodPressure': [72, 66, 64, 66, 40, 74, 50, 0, 70, 96, 80, 78, 84, 92, 68, 76, 86, 74, 74, 84],
    'SkinThickness': [35, 29, 0, 23, 35, 0, 32, 0, 45, 0, 40, 48, 0, 0, 30, 35, 30, 20, 35, 47],
    'Insulin': [0, 0, 0, 94, 168, 0, 88, 0, 543, 0, 0, 110, 0, 0, 130, 105, 100, 0, 0, 230],
    'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3, 30.5, 0.0, 31.2, 36.6, 20.3, 31.8, 30.4, 30.7, 30.4, 29.3, 40.2, 45.8],
    'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288, 0.201, 0.248, 0.134, 0.158, 0.232, 0.231, 0.395, 0.147, 0.191, 0.323, 0.396, 0.517, 0.245, 0.441, 0.465],
    'Age': [50, 31, 32, 21, 33, 30, 26, 29, 53, 54, 26, 30, 23, 28, 29, 21, 40, 38, 51, 35],
    'Outcome': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]
}
df_diabetes = pd.DataFrame(data)

# --- 2. Pra-pemrosesan Data (Sama seperti sebelumnya) ---
# Mengganti nilai 0 yang tidak valid dengan NaN
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero:
    df_diabetes[col] = df_diabetes[col].replace(0, np.nan)

# Mengisi NaN dengan nilai rata-rata (mean)
for col in cols_with_zero:
    df_diabetes[col] = df_diabetes[col].fillna(df_diabetes[col].mean())

# Memisahkan fitur (X) dan target (y)
X = df_diabetes.drop('Outcome', axis=1)
y = df_diabetes['Outcome']

# Scaling Fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Membagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)

# --- 3. Pelatihan Model (Sama seperti sebelumnya) ---
model_diabetes = LogisticRegression(random_state=42)
model_diabetes.fit(X_train, y_train)

# --- 4. Simpan Model dan Scaler ---
# Menyimpan model yang sudah dilatih
joblib.dump(model_diabetes, 'diabetes_model.pkl')
# Menyimpan scaler yang sudah dilatih (penting untuk memproses input baru)
joblib.dump(scaler, 'scaler.pkl')

print("Model dan Scaler telah disimpan sebagai diabetes_model.pkl dan scaler.pkl di direktori saat ini.")
