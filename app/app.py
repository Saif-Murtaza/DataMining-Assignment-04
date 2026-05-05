
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Load artefacts ─────────────────────────────────────────────────────────────
model        = joblib.load("model.pkl")
scaler       = joblib.load("scaler.pkl")
with open("fit_cats.json") as f: fit_cats = json.load(f)
with open("feature_cols.json") as f: FEATURE_COLS = json.load(f)
with open("sample_patient.json") as f: SAMPLE = json.load(f)

CONT_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CardioAI - Heart Disease Screening", layout="wide")

st.title("CardioAI - Heart Disease Screening Dashboard")
st.markdown("*Powered by ML model trained on UCI dataset*")
st.divider()

# ── Sidebar input ──────────────────────────────────────────────────────────────
st.sidebar.header("Patient Input")

def sidebar_input(label, key, min_val, max_val, step=1):
    val = SAMPLE.get(key, min_val)
    return st.sidebar.number_input(
        label,
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(val),
        step=float(step),
        key=key
    )

age      = sidebar_input("Age", "age", 20, 80, 1)
sex      = st.sidebar.selectbox("Sex", [0, 1])
cp       = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = sidebar_input("Resting Blood Pressure", "trestbps", 80, 200, 1)
chol     = sidebar_input("Cholesterol", "chol", 100, 600, 1)
fbs      = st.sidebar.selectbox("Fasting Blood Sugar", [0, 1])
restecg  = st.sidebar.selectbox("Resting ECG", [0, 1, 2])
thalach  = sidebar_input("Max Heart Rate", "thalach", 70, 210, 1)
exang    = st.sidebar.selectbox("Exercise Angina", [0, 1])
oldpeak  = sidebar_input("ST Depression", "oldpeak", 0.0, 7.0, 0.1)
slope    = st.sidebar.selectbox("Slope", [0, 1, 2])
ca       = sidebar_input("Vessels", "ca", 0, 3, 1)
thal     = st.sidebar.selectbox("Thal", [1, 2, 3])

predict_btn = st.sidebar.button("Predict")

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_input(values_dict):
    df = pd.DataFrame([values_dict])

    for col, cats in fit_cats.items():
        for c in cats:
            df[f"{col}_{c}"] = (df[col] == c).astype(int)
        df.drop(columns=[col], inplace=True)

    df[CONT_COLS] = scaler.transform(df[CONT_COLS])

    for fc in FEATURE_COLS:
        if fc not in df.columns:
            df[fc] = 0

    return df[FEATURE_COLS]

# ── Output ─────────────────────────────────────────────────────────────────────
if predict_btn:
    raw_input = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg,
        "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }

    X_input = preprocess_input(raw_input)

    proba = float(model.predict_proba(X_input)[0, 1])
    pred = int(proba >= 0.5)

    if pred == 1:
        st.error("DISEASE PRESENT")
    else:
        st.success("NO DISEASE DETECTED")

    st.metric("Risk Probability", f"{proba*100:.1f}%")

else:
    st.info("Enter patient details and click Predict")

st.divider()
st.caption("Data Mining Assignment | Saif Murtaza | 23i-2588")
