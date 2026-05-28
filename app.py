import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Healthcare CKD Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("ckd_model.pkl", "rb"))

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp {
    background-color: #0B1120;
    color: white;
}

.main-title {
    font-size: 48px;
    font-weight: bold;
    color: white;
}

.sub-text {
    color: #A0AEC0;
    font-size: 18px;
}

.metric-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 0px 12px rgba(0,0,0,0.5);
}

.result-high {
    background-color: #4B1D1D;
    padding: 25px;
    border-radius: 15px;
    color: #FF6B6B;
    font-size: 28px;
    font-weight: bold;
    text-align: center;
}

.result-low {
    background-color: #123524;
    padding: 25px;
    border-radius: 15px;
    color: #4DFF91;
    font-size: 28px;
    font-weight: bold;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🩺 Patient Filters")

st.sidebar.write("Adjust healthcare parameters")

age = st.sidebar.slider(
    "Age",
    1,
    100,
    30
)

bp = st.sidebar.slider(
    "Blood Pressure",
    50,
    180,
    80
)

sg = st.sidebar.slider(
    "Specific Gravity",
    1.000,
    1.030,
    1.020
)

albumin = st.sidebar.slider(
    "Albumin",
    0,
    5,
    0
)

sugar = st.sidebar.slider(
    "Sugar",
    0,
    5,
    0
)

creatinine = st.sidebar.slider(
    "Serum Creatinine",
    0.1,
    15.0,
    1.0
)

hemoglobin = st.sidebar.slider(
    "Hemoglobin",
    3.0,
    18.0,
    15.0
)

# ---------------- MAIN TITLE ----------------
st.markdown(
    '<p class="main-title">🏥 Healthcare CKD Prediction Dashboard</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-text">Machine Learning Based Chronic Kidney Disease Prediction System</p>',
    unsafe_allow_html=True
)

st.write("")

# ---------------- METRIC CARDS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Age</h3>
        <h2>{age}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Blood Pressure</h3>
        <h2>{bp}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Hemoglobin</h3>
        <h2>{hemoglobin}</h2>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# ---------------- INPUT DATA ----------------
input_data = np.array([[
    age,
    bp,
    sg,
    albumin,
    sugar,
    creatinine,
    hemoglobin
]])

# ---------------- PREDICTION BUTTON ----------------
if st.button("🔍 Predict CKD"):

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    risk_score = probability[0][1] * 100

    st.write("")
    st.subheader("Prediction Result")

    # ---------------- LOW RISK ----------------
    if prediction[0] == 0:

        st.markdown(f"""
        <div class="result-low">
        ✅ LOW RISK OF CKD <br><br>
        Probability: {100-risk_score:.2f}%
        </div>
        """, unsafe_allow_html=True)

    # ---------------- HIGH RISK ----------------
    else:

        st.markdown(f"""
        <div class="result-high">
        ⚠️ HIGH RISK OF CKD <br><br>
        Probability: {risk_score:.2f}%
        </div>
        """, unsafe_allow_html=True)

    # ---------------- PATIENT SUMMARY ----------------
    st.write("")
    st.subheader("📋 Patient Summary")

    patient_df = pd.DataFrame({
        "Feature": [
            "Age",
            "Blood Pressure",
            "Specific Gravity",
            "Albumin",
            "Sugar",
            "Creatinine",
            "Hemoglobin"
        ],

        "Value": [
            age,
            bp,
            sg,
            albumin,
            sugar,
            creatinine,
            hemoglobin
        ]
    })

    st.dataframe(patient_df, use_container_width=True)

    # ---------------- CHART ----------------
    st.write("")
    st.subheader("📊 Health Indicators")

    chart_df = pd.DataFrame({
        "Feature": [
            "Blood Pressure",
            "Creatinine",
            "Hemoglobin"
        ],

        "Value": [
            bp,
            creatinine,
            hemoglobin
        ]
    })

    st.bar_chart(chart_df.set_index("Feature"))

# ---------------- FOOTER ----------------
st.write("")
st.markdown("---")
st.caption("Developed using Streamlit & Machine Learning")