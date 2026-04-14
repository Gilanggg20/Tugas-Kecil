import pickle
import pandas as pd
import streamlit as st

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Prediksi Risiko Kredit",
    page_icon="💳",
    layout="centered"
)

# ============================================================
# LOAD MODEL DAN METADATA
# ============================================================
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

feature_columns = metadata["feature_columns"]
best_model_name = metadata["best_model_name"]

# ============================================================
# FUNGSI BANTU
# ============================================================
risk_descriptions = {
    "Very Low Risk": "Risiko kredit sangat rendah. Calon peminjam tergolong sangat aman.",
    "Low Risk": "Risiko kredit rendah. Kemungkinan gagal bayar relatif kecil.",
    "Moderate Risk": "Risiko kredit sedang. Perlu pertimbangan lebih lanjut sebelum persetujuan.",
    "High Risk": "Risiko kredit tinggi. Perlu perhatian lebih dalam proses analisis kredit.",
    "Very High Risk": "Risiko kredit sangat tinggi. Potensi gagal bayar besar dan perlu kehati-hatian."
}

risk_colors = {
    "Very Low Risk": "#16a34a",
    "Low Risk": "#22c55e",
    "Moderate Risk": "#eab308",
    "High Risk": "#f97316",
    "Very High Risk": "#dc2626"
}

def build_prediction_card(prediction: str):
    color = risk_colors.get(prediction, "#2563eb")
    description = risk_descriptions.get(prediction, "Tidak ada deskripsi.")
    card_html = f"""
    <div style="
        background-color: {color}20;
        border-left: 8px solid {color};
        padding: 18px;
        border-radius: 12px;
        margin-top: 10px;
        margin-bottom: 15px;
    ">
        <h3 style="margin: 0; color: {color};">Hasil Prediksi: {prediction}</h3>
        <p style="margin-top: 8px; font-size: 15px;">{description}</p>
    </div>
    """
    return card_html

# ============================================================
# HEADER
# ============================================================
st.title("💳 Prediksi Risiko Kredit")
st.markdown("### Sistem Prediksi Risiko Kredit Berbasis Machine Learning")
st.caption(f"Model yang digunakan: {best_model_name}")

st.write(
    """
Aplikasi ini digunakan untuk memprediksi tingkat risiko kredit calon peminjam
berdasarkan data finansial yang dimasukkan pengguna.
"""
)

# ============================================================
# FORM INPUT
# ============================================================
st.subheader("Input Data Calon Peminjam")

with st.form("credit_form"):
    col1, col2 = st.columns(2)

    with col1:
        person_age = st.number_input("Usia", min_value=18, max_value=100, value=30)
        person_income = st.number_input("Pendapatan Tahunan", min_value=0.0, value=50000000.0, step=1000000.0)
        person_home_ownership = st.selectbox(
            "Status Kepemilikan Rumah",
            ["RENT", "OWN", "MORTGAGE", "OTHER"]
        )
        person_emp_length = st.number_input("Lama Bekerja (tahun)", min_value=0.0, value=5.0, step=1.0)
        loan_intent = st.selectbox(
            "Tujuan Pinjaman",
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
        )

    with col2:
        loan_amnt = st.number_input("Jumlah Pinjaman", min_value=0.0, value=10000000.0, step=500000.0)
        loan_int_rate = st.number_input("Suku Bunga Pinjaman", min_value=0.0, value=12.0, step=0.1)
        loan_percent_income = st.number_input("Rasio Pinjaman terhadap Pendapatan", min_value=0.0, value=0.25, step=0.01)
        cb_person_default_on_file = st.selectbox("Riwayat Default Sebelumnya", ["Y", "N"])
        cb_person_cred_hist_length = st.number_input("Lama Riwayat Kredit (tahun)", min_value=0.0, value=5.0, step=1.0)

    submitted = st.form_submit_button("🔍 Prediksi Risiko")

# ============================================================
# PREDIKSI
# ============================================================
if submitted:
    input_data = pd.DataFrame([{
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length
    }])

    input_data = input_data.reindex(columns=feature_columns)

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    class_names = model.named_steps["model"].classes_

    # Hasil utama
    st.markdown(build_prediction_card(prediction), unsafe_allow_html=True)

    # Ringkasan input
    with st.expander("Lihat Ringkasan Input"):
        st.dataframe(input_data, use_container_width=True)

    # Probabilitas
    probability_df = pd.DataFrame({
        "Risk Level": class_names,
        "Probability": probabilities
    }).sort_values(by="Probability", ascending=False)

    st.subheader("📊 Probabilitas Prediksi")
    st.dataframe(
        probability_df.assign(
            Probability=lambda df: (df["Probability"] * 100).round(2).astype(str) + "%"
        ),
        use_container_width=True
    )

    st.subheader("Visualisasi Keyakinan Model")
    for _, row in probability_df.iterrows():
        label = row["Risk Level"]
        prob = float(row["Probability"])
        st.write(f"**{label}** — {prob * 100:.2f}%")
        st.progress(prob)

    # Insight singkat
    top_probability = probability_df.iloc[0]["Probability"] * 100
    st.info(
        f"Model paling yakin bahwa calon peminjam berada pada kategori **{prediction}** "
        f"dengan tingkat keyakinan sekitar **{top_probability:.2f}%**."
    )
