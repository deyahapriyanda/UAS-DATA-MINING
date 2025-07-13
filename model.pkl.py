import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # Untuk np.nan

# --- 1. Pengumpulan dan Pemahaman Data ---
# Dataset ini tersedia secara online. Untuk demonstrasi, kita bisa membuatnya secara manual
# atau mengasumsikan kita memuatnya dari file CSV.
# Biasanya, Anda akan memuatnya dengan:
# df_diabetes = pd.read_csv('nama_file_dataset_diabetes.csv')

# Simulasi Dataset Pima Indians Diabetes
# Source: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
data = {
    'Pregnancies': [6, 1, 8, 1, 0, 5, 3, 10, 2, 8],
    'Glucose': [148, 85, 183, 89, 137, 116, 78, 115, 197, 125],
    'BloodPressure': [72, 66, 64, 66, 40, 74, 50, 0, 70, 96], # BloodPressure 0 bisa jadi missing value
    'SkinThickness': [35, 29, 0, 23, 35, 0, 32, 0, 45, 0], # SkinThickness 0 bisa jadi missing value
    'Insulin': [0, 0, 0, 94, 168, 0, 88, 0, 543, 0], # Insulin 0 bisa jadi missing value
    'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3, 30.5, 0.0], # BMI 0 bisa jadi missing value
    'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288, 0.201, 0.248, 0.134, 0.158, 0.232],
    'Age': [50, 31, 32, 21, 33, 30, 26, 29, 53, 54],
    'Outcome': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1] # 1 = Diabetes, 0 = Non-Diabetes
}
df_diabetes = pd.DataFrame(data)

print("--- 1. Contoh Data Awal (Pima Indians Diabetes Dataset) ---")
print(df_diabetes.head())
print(f"\nUkuran dataset: {df_diabetes.shape[0]} baris, {df_diabetes.shape[1]} kolom")
print("\nInformasi tipe data:")
print(df_diabetes.info())
print("\n" + "="*70 + "\n")

# --- 2. Pra-pemrosesan Data ---

# Menangani nilai 0 yang tidak valid (missing values)
# Beberapa kolom seperti Glucose, BloodPressure, SkinThickness, Insulin, BMI
# tidak mungkin bernilai 0 dalam konteks medis yang berarti.
# Kita akan mengganti nilai 0 ini dengan nilai NaN (Not a Number)
# dan kemudian mengisi NaN dengan nilai rata-rata (mean) dari kolom tersebut.

cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero:
    df_diabetes[col] = df_diabetes[col].replace(0, np.nan)

print("--- 2a. Data Setelah Mengganti '0' dengan NaN ---")
print(df_diabetes.isnull().sum()) # Cek jumlah NaN per kolom
print("\n")

# Mengisi NaN dengan nilai rata-rata (imputasi mean)
for col in cols_with_zero:
    df_diabetes[col] = df_diabetes[col].fillna(df_diabetes[col].mean())

print("--- 2b. Data Setelah Imputasi Mean (Mengisi NaN dengan Rata-rata) ---")
print(df_diabetes.isnull().sum()) # Pastikan sudah tidak ada NaN
print("\n")

# Memisahkan fitur (X) dan target (y)
X = df_diabetes.drop('Outcome', axis=1) # Semua kolom kecuali 'Outcome'
y = df_diabetes['Outcome']              # Kolom 'Outcome'

# Scaling Fitur: Menstandardisasi fitur numerik
# Ini penting untuk model seperti Regresi Logistik agar semua fitur memiliki skala yang sama,
# mencegah fitur dengan rentang nilai besar mendominasi model.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Mengubah X menjadi array numpy yang sudah diskalakan
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns) # Kembalikan ke DataFrame untuk kemudahan

print("--- 2c. Data Fitur Setelah Scaling (Standardisasi) ---")
print(X_scaled_df.head())
print("\n")

# Membagi data menjadi set pelatihan dan set pengujian
# 80% data untuk training, 20% untuk testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)
# stratify=y penting untuk klasifikasi biner, memastikan proporsi kelas di train/test sama

print(f"Jumlah data pelatihan: {len(X_train)} sampel")
print(f"Jumlah data pengujian: {len(X_test)} sampel")
print(f"Proporsi kelas 'Outcome' di y_train:\n{y_train.value_counts(normalize=True)}")
print(f"Proporsi kelas 'Outcome' di y_test:\n{y_test.value_counts(normalize=True)}")
print("\n" + "="*70 + "\n")

# --- 3. Pelatihan Model ---
# Menggunakan Regresi Logistik sebagai model klasifikasi

model_diabetes = LogisticRegression(random_state=42)
model_diabetes.fit(X_train, y_train)

print("Model Regresi Logistik untuk prediksi diabetes berhasil dilatih.\n")
print("\n" + "="*70 + "\n")

# --- 4. Evaluasi Model ---
# Mengevaluasi performa model pada data pengujian

y_pred = model_diabetes.predict(X_test)
y_pred_proba = model_diabetes.predict_proba(X_test)[:, 1] # Probabilitas kelas positif (diabetes)

# Akurasi Model
akurasi = accuracy_score(y_test, y_pred)
print(f"Akurasi Model: {akurasi:.4f}\n")

# Laporan Klasifikasi (Precision, Recall, F1-score)
print("--- Laporan Klasifikasi ---")
# Outcome 0 = Non-Diabetes, 1 = Diabetes
print(classification_report(y_test, y_pred, target_names=['Non-Diabetes', 'Diabetes']))
print("\n")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("--- Confusion Matrix ---")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Diprediksi Non-Diabetes', 'Diprediksi Diabetes'],
            yticklabels=['Aktual Non-Diabetes', 'Aktual Diabetes'])
plt.title('Confusion Matrix Prediksi Diabetes')
plt.ylabel('Aktual')
plt.xlabel('Prediksi')
plt.show()

print("\n" + "="*70 + "\n")

# --- 5. Contoh Prediksi untuk Data Baru ---
# Misalkan ada data pasien baru:
# Pregnancies: 2
# Glucose: 120
# BloodPressure: 70
# SkinThickness: 30
# Insulin: 100
# BMI: 30.5
# DiabetesPedigreeFunction: 0.4
# Age: 35

new_patient_data = pd.DataFrame({
    'Pregnancies': [2],
    'Glucose': [120],
    'BloodPressure': [70],
    'SkinThickness': [30],
    'Insulin': [100],
    'BMI': [30.5],
    'DiabetesPedigreeFunction': [0.4],
    'Age': [35]
})

# Penting: Data baru juga harus melalui proses scaling yang sama!
new_patient_scaled = scaler.transform(new_patient_data) # Gunakan scaler yang sudah di-fit
new_patient_scaled_df = pd.DataFrame(new_patient_scaled, columns=X.columns)

# Prediksi
prediction = model_diabetes.predict(new_patient_scaled_df)
prediction_proba = model_diabetes.predict_proba(new_patient_scaled_df)

status = "Diabetes" if prediction[0] == 1 else "Non-Diabetes"
prob_diabetes = prediction_proba[0, 1] * 100

print(f"--- Prediksi Risiko Diabetes untuk Pasien Baru ---")
print(f"Data Pasien: \n{new_patient_data.iloc[0]}")
print(f"\nPrediksi Status: **{status}**")
print(f"Probabilitas Diabetes: **{prob_diabetes:.2f}%**")
print("\n" + "="*70 + "\n")