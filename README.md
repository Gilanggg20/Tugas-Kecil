# 💳 Klasifikasi Multi-Level Risiko Kredit

Project ini bertujuan untuk mengklasifikasikan tingkat risiko kredit calon peminjam menggunakan Machine Learning dengan pendekatan **Cost-Sensitive Learning**.

---

## 🎯 Tujuan
Membangun model prediksi risiko kredit untuk membantu menentukan tingkat kelayakan peminjam berdasarkan data finansial.

---

## 📊 Dataset
Dataset yang digunakan adalah dataset finansial yang berisi informasi seperti:
- Usia
- Pendapatan
- Jumlah pinjaman
- Suku bunga
- Rasio pinjaman terhadap pendapatan
- Riwayat kredit

Target utama berasal dari **loan_grade** yang kemudian diubah menjadi:

| Loan Grade | Risk Level |
|------------|-----------|
| A | Very Low Risk |
| B | Low Risk |
| C | Moderate Risk |
| D | High Risk |
| E, F, G | Very High Risk |

---

## 🤖 Algoritma yang Digunakan
- CatBoost
- XGBoost
- LightGBM

---

## 🏆 Model Terbaik
Model terbaik yang diperoleh adalah:
**LightGBM (hasil hyperparameter tuning)**

---

## ⚙️ Metode yang Digunakan
- Data Cleaning (handling missing values & outlier)
- Feature Engineering
- Encoding & Preprocessing (Pipeline)
- Handling Imbalanced Data (Undersampling)
- Hyperparameter Tuning (RandomizedSearchCV)
- Cost-Sensitive Learning
- Model Evaluation

---

## 📈 Evaluasi Model
Model dievaluasi menggunakan:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 🌐 Deployment
Aplikasi web dibuat menggunakan:
- **Streamlit**
- **pyngrok** (untuk hosting sementara di Google Colab)

Aplikasi dapat menerima input user dan memprediksi:
- Risk Level
- Probabilitas tiap kelas
