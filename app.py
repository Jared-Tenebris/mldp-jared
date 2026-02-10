# app.py
import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import sklearn

st.set_page_config(
    page_title="Cardiovascular Health Risk Screener",
    page_icon="❤️",
    layout="wide"
)

# Put your model here later:
# mldp-jared/
#   app.py
#   models/
#     cardiovascular_health_model.pkl
MODEL_PATH = os.path.join("models", "cardiovascular_health_model.pkl")

# -------------------------
# Known categories (stable one-hot alignment)
# -------------------------
CATS = {
    "General_Health": ["Excellent", "Fair", "Good", "Poor", "Very Good"],
    "Checkup": [
        "5 or more years ago",
        "Never",
        "Within the past 2 years",
        "Within the past 5 years",
        "Within the past year",
    ],
    "Exercise": ["No", "Yes"],
    "Skin_Cancer": ["No", "Yes"],
    "Other_Cancer": ["No", "Yes"],
    "Depression": ["No", "Yes"],
    "Diabetes": ["No", "Yes"],
    "Arthritis": ["No", "Yes"],
    "Sex": ["Female", "Male"],
    "Age_Category": [
        "18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
        "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+",
    ],
    "Smoking_History": ["No", "Yes"],
}

NUM_COLS = [
    "Height_(cm)", "Weight_(kg)", "BMI",
    "Alcohol_Consumption", "Fruit_Consumption",
    "Green_Vegetables_Consumption", "FriedPotato_Consumption",
]
CAT_COLS = list(CATS.keys())

def expected_dummy_columns():
    cols = []
    cols.extend(NUM_COLS)
    for col in CAT_COLS:
        for cat in CATS[col][1:]:  # drop_first=True
            cols.append(f"{col}_{cat}")
    return cols

EXPECTED_COLS = expected_dummy_columns()

# -------------------------
# Helpers (safe; never crash UI)
# -------------------------
def show_env_versions():
    st.caption(
        f"Runtime versions — Python: {sys.version.split()[0]} | "
        f"NumPy: {np.__version__} | pandas: {pd.__version__} | "
        f"scikit-learn: {sklearn.__version__} | joblib: {joblib.__version__}"
    )

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def try_load_model(path: str):
    """
    Non-fatal model loader:
    Returns (model_or_None, error_string_or_None).
    UI will ALWAYS load.
    """
    if not os.path.exists(path):
        return None, f"Model file not found at: {path}"
    try:
        m = joblib.load(path)
        return m, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

def validate_inputs(h_cm, w_kg, bmi, alcohol, fruit, veg, fried):
    errors = []
    if h_cm is None or h_cm <= 0:
        errors.append("Height must be a positive number.")
    if w_kg is None or w_kg <= 0:
        errors.append("Weight must be a positive number.")
    if bmi is None or bmi <= 0:
        errors.append("BMI must be a positive number.")
    for name, val in [
        ("Alcohol consumption", alcohol),
        ("Fruit consumption", fruit),
        ("Green vegetables consumption", veg),
        ("Fried potato consumption", fried),
    ]:
        if val is None or val < 0:
            errors.append(f"{name} must be 0 or more.")
    return errors

def encode_like_training(raw_df: pd.DataFrame) -> pd.DataFrame:
    for col in CAT_COLS:
        raw_df[col] = pd.Categorical(raw_df[col], categories=CATS[col], ordered=True)
    encoded = pd.get_dummies(raw_df, drop_first=True)
    aligned = encoded.reindex(columns=EXPECTED_COLS, fill_value=0)
    return aligned.astype(float)

def predict_proba_1(model, X: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if isinstance(proba, np.ndarray) and proba.shape[1] >= 2:
            return float(proba[0, 1])
        return float(proba[0])
    pred = int(model.predict(X)[0])
    return 1.0 if pred == 1 else 0.0

# -------------------------
# UI (always loads)
# -------------------------
st.title("❤️ Cardiovascular Health Risk Screener")
st.caption("Risk screening only — not a medical diagnosis.")
show_env_versions()

model, model_err = try_load_model(MODEL_PATH)

with st.expander("Important note", expanded=True):
    st.write(
        "- This app estimates cardiovascular risk using a trained ML model.\n"
        "- It is **not** a diagnosis.\n"
        "- If you have concerns, seek advice from a qualified healthcare professional."
    )

# Model status banner (won't block UI)
if model is None:
    st.warning("Model is not loaded yet — UI is working, but prediction is disabled for now.")
    with st.expander("Model load details"):
        st.code(model_err)

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.subheader("1) Enter your details")

    GENERAL_HEALTH_UI = ["Excellent", "Very Good", "Good", "Fair", "Poor"]
    CHECKUP_UI = ["Within the past year", "Within the past 2 years", "Within the past 5 years", "5 or more years ago", "Never"]
    YES_NO = ["No", "Yes"]
    SEX_UI = ["Female", "Male"]
    AGE_UI = CATS["Age_Category"]

    with st.form("risk_form", clear_on_submit=False):
        c1, c2 = st.columns(2)

        with c1:
            general_health = st.selectbox("General health", GENERAL_HEALTH_UI, index=2)
            checkup = st.selectbox("Last medical checkup", CHECKUP_UI, index=0)
            age_category = st.selectbox("Age group", AGE_UI, index=0)
            sex = st.selectbox("Sex", SEX_UI, index=0)

        with c2:
            exercise = st.selectbox("Do you exercise?", YES_NO, index=1)
            smoking_history = st.selectbox("Smoking history", YES_NO, index=0)
            diabetes = st.selectbox("Diabetes", YES_NO, index=0)
            depression = st.selectbox("Depression", YES_NO, index=0)

        c3, c4 = st.columns(2)
        with c3:
            skin_cancer = st.selectbox("Skin cancer", YES_NO, index=0)
            other_cancer = st.selectbox("Other cancer", YES_NO, index=0)
        with c4:
            arthritis = st.selectbox("Arthritis", YES_NO, index=0)

        st.markdown("---")
        st.subheader("Body measurements")

        m1, m2, m3 = st.columns(3)
        with m1:
            height_cm_in = st.text_input("Height (cm)", value="170")
        with m2:
            weight_kg_in = st.text_input("Weight (kg)", value="70")
        with m3:
            bmi_mode = st.radio("BMI", ["Auto-calculate", "Enter manually"], horizontal=True, index=0)

        height_cm = safe_float(height_cm_in)
        weight_kg = safe_float(weight_kg_in)

        if bmi_mode == "Auto-calculate" and height_cm and weight_kg and height_cm > 0:
            bmi_val = weight_kg / ((height_cm / 100.0) ** 2)
            st.text_input("BMI (auto)", value=f"{bmi_val:.2f}", disabled=True)
            bmi = bmi_val
        else:
            bmi_in = st.text_input("BMI", value="24.2")
            bmi = safe_float(bmi_in)

        st.markdown("---")
        st.subheader("Diet & lifestyle (numbers)")

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            alcohol_in = st.text_input("Alcohol consumption", value="0")
        with d2:
            fruit_in = st.text_input("Fruit consumption", value="1")
        with d3:
            veg_in = st.text_input("Green vegetables consumption", value="1")
        with d4:
            fried_in = st.text_input("Fried potato consumption", value="0")

        alcohol = safe_float(alcohol_in)
        fruit = safe_float(fruit_in)
        veg = safe_float(veg_in)
        fried = safe_float(fried_in)

        submitted = st.form_submit_button("Predict risk")

with right:
    st.subheader("2) Results")

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if submitted:
        errs = validate_inputs(height_cm, weight_kg, bmi, alcohol, fruit, veg, fried)

        if errs:
            st.error("Please fix these before predicting:")
            for e in errs:
                st.write(f"- {e}")

        elif model is None:
            st.error("Prediction disabled (model not loaded). UI is working — we will fix model later.")

        else:
            raw_df = pd.DataFrame([{
                "General_Health": general_health,
                "Checkup": checkup,
                "Exercise": exercise,
                "Skin_Cancer": skin_cancer,
                "Other_Cancer": other_cancer,
                "Depression": depression,
                "Diabetes": diabetes,
                "Arthritis": arthritis,
                "Sex": sex,
                "Age_Category": age_category,
                "Smoking_History": smoking_history,
                "Height_(cm)": float(height_cm),
                "Weight_(kg)": float(weight_kg),
                "BMI": float(bmi),
                "Alcohol_Consumption": float(alcohol),
                "Fruit_Consumption": float(fruit),
                "Green_Vegetables_Consumption": float(veg),
                "FriedPotato_Consumption": float(fried),
            }])

            try:
                X_input = encode_like_training(raw_df)
                prob = predict_proba_1(model, X_input)
                label = "Higher risk" if prob >= 0.5 else "Lower risk"
                st.session_state.last_result = {"prob": prob, "label": label, "X": X_input}
            except Exception as e:
                st.error("Prediction failed (we will fix model/features later).")
                st.exception(e)

    res = st.session_state.last_result
    if res is None:
        st.info("Fill in the form and click **Predict risk** to see results.")
    else:
        prob = res["prob"]
        label = res["label"]

        st.metric("Risk level", label)
        st.write(f"Estimated probability: **{prob*100:.1f}%**")
        st.progress(min(max(prob, 0.0), 1.0))

        with st.expander("Show model input (debug)"):
            st.dataframe(res["X"], use_container_width=True)

st.markdown("---")
st.caption(f"Model path: {MODEL_PATH} • App: app.py")