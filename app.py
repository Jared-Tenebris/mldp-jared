# app.py
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# -------------------------
# Config
# -------------------------
st.set_page_config(
    page_title="Cardiovascular Health Risk Screener",
    page_icon="❤️",
    layout="wide",
)

MODEL_PATH = "cardiovascular_health_model.pkl"

# -------------------------
# Helpers
# -------------------------
def load_model(path: str):
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}\n\nMake sure '{MODEL_PATH}' is in the same folder as app.py.")
        st.stop()
    try:
        return joblib.load(path)
    except Exception as e:
        st.error("Failed to load the model. The file may be corrupted or incompatible.")
        st.exception(e)
        st.stop()

def get_expected_feature_columns(model):
    # Best case: sklearn estimators fitted on a DataFrame store feature_names_in_
    cols = getattr(model, "feature_names_in_", None)
    if cols is not None:
        return list(cols)

    # Fallback: some models store n_features_in_ but not names
    st.error(
        "This model does not contain feature names (feature_names_in_ missing). "
        "Re-train using a pandas DataFrame so feature names are preserved."
    )
    st.stop()

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def validate_inputs(h_cm, w_kg, bmi, alcohol, fruit, veg, fried):
    errors = []

    if h_cm is None or h_cm <= 0:
        errors.append("Height must be a positive number.")
    elif h_cm < 100 or h_cm > 230:
        errors.append("Height looks unusual. Enter a value between 100 and 230 cm.")

    if w_kg is None or w_kg <= 0:
        errors.append("Weight must be a positive number.")
    elif w_kg < 25 or w_kg > 300:
        errors.append("Weight looks unusual. Enter a value between 25 and 300 kg.")

    if bmi is None or bmi <= 0:
        errors.append("BMI must be a positive number.")
    elif bmi < 10 or bmi > 60:
        errors.append("BMI looks unusual. Enter a value between 10 and 60.")

    # Consumption fields: allow 0 and up
    for name, val, hi in [
        ("Alcohol consumption", alcohol, 200),
        ("Fruit consumption", fruit, 50),
        ("Green vegetables consumption", veg, 50),
        ("Fried potato consumption", fried, 50),
    ]:
        if val is None or val < 0:
            errors.append(f"{name} must be 0 or more.")
        elif val > hi:
            errors.append(f"{name} looks too high. Enter a value less than or equal to {hi}.")

    return errors

def build_raw_input_row(
    general_health,
    checkup,
    exercise,
    skin_cancer,
    other_cancer,
    depression,
    diabetes,
    arthritis,
    sex,
    age_category,
    smoking_history,
    height_cm,
    weight_kg,
    bmi,
    alcohol,
    fruit,
    veg,
    fried_potato,
):
    # Column names must match the original dataset BEFORE get_dummies
    return pd.DataFrame([{
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
        "Height_(cm)": height_cm,
        "Weight_(kg)": weight_kg,
        "BMI": bmi,
        "Alcohol_Consumption": alcohol,
        "Fruit_Consumption": fruit,
        "Green_Vegetables_Consumption": veg,
        "FriedPotato_Consumption": fried_potato,
    }])

def encode_like_training(raw_df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    # Training used: pd.get_dummies(X, drop_first=True) on ALL predictors
    encoded = pd.get_dummies(raw_df, drop_first=True)

    # Align to model expected columns (missing -> 0)
    aligned = encoded.reindex(columns=expected_cols, fill_value=0)

    # Ensure numeric dtype
    return aligned.astype(float)

# -------------------------
# Load model & expected cols
# -------------------------
model = load_model(MODEL_PATH)
expected_cols = get_expected_feature_columns(model)

# -------------------------
# UI
# -------------------------
st.title("❤️ Cardiovascular Health Risk Screener")
st.caption("A simple risk screening tool. Not a medical diagnosis.")

with st.expander("Important note", expanded=True):
    st.write(
        "- This app estimates risk based on a machine learning model.\n"
        "- It is **not** a diagnosis.\n"
        "- If you are worried about symptoms or health, talk to a qualified healthcare professional."
    )

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.subheader("1) Enter your details")

    GENERAL_HEALTH = ["Excellent", "Very Good", "Good", "Fair", "Poor"]
    CHECKUP = [
        "Within the past year",
        "Within the past 2 years",
        "Within the past 5 years",
        "5 or more years ago",
        "Never",
    ]
    YES_NO = ["No", "Yes"]
    SEX = ["Female", "Male"]
    AGE_CATEGORY = [
        "18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
        "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"
    ]

    with st.form("risk_form", clear_on_submit=False):
        c1, c2 = st.columns(2)

        with c1:
            general_health = st.selectbox("General health", GENERAL_HEALTH, index=2)
            checkup = st.selectbox("Last medical checkup", CHECKUP, index=0)
            age_category = st.selectbox("Age group", AGE_CATEGORY, index=0)
            sex = st.selectbox("Sex", SEX, index=0)

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
            bmi_in = st.text_input("BMI (auto)", value=f"{bmi_val:.2f}", disabled=True)
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

    if "submitted_flag" not in st.session_state:
        st.session_state.submitted_flag = False

    # When form submits, Streamlit reruns; detect with the local variable
    try:
        submitted
    except NameError:
        submitted = False

    if submitted:
        errs = validate_inputs(height_cm, weight_kg, bmi, alcohol, fruit, veg, fried)
        if errs:
            st.error("Please fix these before predicting:")
            for e in errs:
                st.write(f"- {e}")
        else:
            raw_df = build_raw_input_row(
                general_health=general_health,
                checkup=checkup,
                exercise=exercise,
                skin_cancer=skin_cancer,
                other_cancer=other_cancer,
                depression=depression,
                diabetes=diabetes,
                arthritis=arthritis,
                sex=sex,
                age_category=age_category,
                smoking_history=smoking_history,
                height_cm=float(height_cm),
                weight_kg=float(weight_kg),
                bmi=float(bmi),
                alcohol=float(alcohol),
                fruit=float(fruit),
                veg=float(veg),
                fried_potato=float(fried),
            )

            try:
                X_input = encode_like_training(raw_df, expected_cols)

                # Predict probability of class 1 (Yes / has heart disease)
                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba(X_input)[0, 1])
                else:
                    # Fallback if no predict_proba (unlikely here)
                    pred = int(model.predict(X_input)[0])
                    prob = 1.0 if pred == 1 else 0.0

                pred_label = "Higher risk" if prob >= 0.5 else "Lower risk"
                st.session_state.last_result = {"prob": prob, "label": pred_label}
            except Exception as e:
                st.error("Prediction failed due to input/feature mismatch.")
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

        st.markdown("---")
        st.subheader("What you can do next")
        st.write(
            "- If your risk looks high, consider a proper health checkup.\n"
            "- Focus on healthy habits (exercise, balanced diet, sleep).\n"
            "- If you have concerning symptoms, seek medical advice."
        )

        with st.expander("Show the processed inputs (for debugging)"):
            # Show the aligned features used by the model
            try:
                st.dataframe(X_input, use_container_width=True)
            except Exception:
                st.write("Processed features unavailable.")

st.markdown("---")
st.caption("Model file: cardiovascular_health_model.pkl | App file: app.py")