# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Tuple

# -------------------------
# Page config + basic theme
# -------------------------
st.set_page_config(
    page_title="Cardiovascular Health Risk Screener",
    page_icon="❤️",
    layout="centered",
)

st.title("❤️ Cardiovascular Health Risk Screener")
st.caption("Risk screening tool (not a medical diagnosis). For educational use in your ML project demo.")

st.info(
    "This app estimates **cardiovascular risk** based on your trained ML model.\n\n"
    "- It is **not** a medical diagnosis.\n"
    "- If someone is worried about their health, they should talk to a qualified healthcare professional."
)

# -------------------------
# Load model
# -------------------------
MODEL_PATH = "cardiovascular_health_model.pkl"

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found.\n\n"
        "✅ Fix: Put **cardiovascular_health_model.pkl** in the SAME folder as this app.py file."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error("Failed to load the model file. The .pkl might be corrupted or saved differently.")
    st.exception(e)
    st.stop()

# -------------------------
# Feature name extraction (robust)
# -------------------------
def get_feature_columns(m) -> List[str]:
    # Best-case: you embedded feature_names_ during export
    if hasattr(m, "feature_names_"):
        cols = list(getattr(m, "feature_names_"))
        if cols:
            return cols

    # Some sklearn objects implement get_feature_names_out
    try:
        if hasattr(m, "get_feature_names_out"):
            cols = list(m.get_feature_names_out())
            if cols:
                return cols
    except Exception:
        pass

    # If pipeline, try transformer steps
    try:
        if hasattr(m, "named_steps"):
            for _step_name, step in m.named_steps.items():
                if hasattr(step, "get_feature_names_out"):
                    cols = list(step.get_feature_names_out())
                    if cols:
                        return cols
    except Exception:
        pass

    return []

feature_columns = get_feature_columns(model)

if not feature_columns:
    st.error(
        "Could not detect the model's input feature names.\n\n"
        "✅ Fix (recommended): when exporting your model, attach feature names like:\n"
        "model.feature_names_ = list(X_train.columns)  # before joblib.dump(...)"
    )
    st.stop()

# -------------------------
# Helpers: build input safely
# -------------------------
def make_empty_input(columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame([np.zeros(len(columns), dtype=float)], columns=columns)

def set_value(df: pd.DataFrame, col: str, value):
    if col in df.columns:
        df.at[0, col] = value

def prettify_label(text: str) -> str:
    return text.replace("_", " ").replace("  ", " ").strip()

def infer_dummy_groups(columns: List[str]) -> Dict[str, List[str]]:
    """
    Groups one-hot columns by prefix using the last underscore split:
      Prefix_CategoryValue
    Example:
      General_Health_Good, General_Health_Fair -> prefix General_Health
    """
    groups: Dict[str, List[str]] = {}
    for c in columns:
        if "_" not in c:
            continue
        prefix, _suffix = c.rsplit("_", 1)
        groups.setdefault(prefix, []).append(c)

    # keep only groups that look like categorical one-hot (>=2 columns)
    groups = {p: sorted(cols) for p, cols in groups.items() if len(cols) >= 2}
    return groups

def is_binary_col(col: str) -> bool:
    # common dummy/binary patterns
    return (
        col.endswith("_Yes")
        or col.endswith("_No")
        or col.endswith("_Positive")
        or col.endswith("_True")
        or col.endswith("_1")
    )

def get_numeric_cols(columns: List[str], dummy_groups: Dict[str, List[str]]) -> List[str]:
    dummy_cols = set()
    for cols in dummy_groups.values():
        dummy_cols.update(cols)

    numeric = []
    for c in columns:
        if c in dummy_cols:
            continue
        if is_binary_col(c):
            continue
        numeric.append(c)
    return sorted(numeric)

def get_single_dummy_binaries(columns: List[str], dummy_groups: Dict[str, List[str]]) -> List[str]:
    dummy_cols = set()
    for cols in dummy_groups.values():
        dummy_cols.update(cols)

    binaries = []
    for c in columns:
        if c in dummy_cols:
            continue
        if is_binary_col(c):
            binaries.append(c)
    return sorted(binaries)

def set_one_hot(df: pd.DataFrame, prefix: str, chosen_value: str, group_cols: List[str], baseline_label: str = "(baseline)"):
    # clear group
    for c in group_cols:
        df.at[0, c] = 0

    # baseline => all zeros
    if chosen_value == baseline_label:
        return

    target = f"{prefix}_{chosen_value}"
    if target in df.columns:
        df.at[0, target] = 1

def safe_float(x, default=None):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return default

# -------------------------
# Auto UI structure
# -------------------------
dummy_groups = infer_dummy_groups(feature_columns)
numeric_cols = get_numeric_cols(feature_columns, dummy_groups)
binary_cols = get_single_dummy_binaries(feature_columns, dummy_groups)

user_input = make_empty_input(feature_columns)

# -------------------------
# Sidebar: help + sanity info
# -------------------------
with st.sidebar:
    st.header("ℹ️ How to use")
    st.write(
        "1) Fill in inputs\n"
        "2) Click **Predict**\n"
        "3) View prediction + probability\n\n"
        "If your model used one-hot encoding with drop_first=True, "
        "each dropdown includes a **(baseline)** option that means “all zeros”."
    )
    st.divider()
    st.write(f"**Model file:** {MODEL_PATH}")
    st.write(f"**Detected features:** {len(feature_columns)}")

# -------------------------
# Input validation rules (safe defaults)
# -------------------------
RANGE_HINTS: Dict[str, Tuple[float, float, float]] = {
    "BMI": (10.0, 60.0, 25.0),
    "Height_(cm)": (120.0, 220.0, 170.0),
    "Weight_(kg)": (30.0, 200.0, 70.0),
    "Sleep_Hours": (0.0, 24.0, 7.0),
    "Physical_Health_Days": (0.0, 30.0, 0.0),
    "Mental_Health_Days": (0.0, 30.0, 0.0),
    "Alcohol_Consumption": (0.0, 100.0, 0.0),
    "Fruit_Consumption": (0.0, 50.0, 0.0),
    "Green_Vegetables_Consumption": (0.0, 50.0, 0.0),
    "FriedPotato_Consumption": (0.0, 50.0, 0.0),
}

def numeric_widget(col: str):
    label = prettify_label(col)
    lo, hi, default = RANGE_HINTS.get(col, (0.0, 100.0, 0.0))

    if "Days" in label and col not in RANGE_HINTS:
        lo, hi, default = 0.0, 30.0, 0.0

    if "Consumption" in label and col not in RANGE_HINTS:
        lo, hi, default = 0.0, 100.0, 0.0

    val = st.number_input(
        label,
        min_value=float(lo),
        max_value=float(hi),
        value=float(default),
        step=1.0 if hi - lo >= 10 else 0.5,
        help=f"Expected range: {lo} to {hi}",
    )
    set_value(user_input, col, float(val))

def binary_widget(col: str):
    label = prettify_label(col)
    display = {0: "No (0)", 1: "Yes (1)"}
    choice = st.selectbox(label, [0, 1], format_func=lambda x: display[x], index=0)
    set_value(user_input, col, int(choice))

def categorical_widget(prefix: str, group_cols: List[str]):
    suffixes = [c.replace(prefix + "_", "") for c in group_cols]
    opts = ["(baseline)"] + suffixes
    chosen = st.selectbox(prettify_label(prefix), opts, index=0)
    set_one_hot(user_input, prefix, chosen, group_cols, baseline_label="(baseline)")

def validate_inputs(df: pd.DataFrame) -> List[str]:
    errors = []

    # Example validation: BMI must be > 0 if present
    if "BMI" in df.columns:
        bmi = safe_float(df.at[0, "BMI"], default=0.0)
        if bmi is None or bmi <= 0:
            errors.append("BMI must be greater than 0.")

    # Sleep hours 0-24 if present
    if "Sleep_Hours" in df.columns:
        sh = safe_float(df.at[0, "Sleep_Hours"], default=0.0)
        if sh is None or sh < 0 or sh > 24:
            errors.append("Sleep Hours must be between 0 and 24.")

    # No NaN / inf anywhere
    if not np.isfinite(df.to_numpy()).all():
        errors.append("Some inputs are invalid (NaN/Infinity). Please re-check your fields.")

    return errors

# -------------------------
# Main form (interactive + safe)
# -------------------------
with st.form("predict_form"):
    st.subheader("🧾 Enter details")

    if numeric_cols:
        st.markdown("#### Numeric fields")
        cols2 = st.columns(2)
        for i, col in enumerate(numeric_cols):
            with cols2[i % 2]:
                numeric_widget(col)

    if dummy_groups:
        st.markdown("#### Categorical fields (dropdowns)")
        for prefix in sorted(dummy_groups.keys()):
            categorical_widget(prefix, dummy_groups[prefix])

    if binary_cols:
        st.markdown("#### Yes/No fields")
        cols2 = st.columns(2)
        for i, col in enumerate(binary_cols):
            with cols2[i % 2]:
                binary_widget(col)

    submitted = st.form_submit_button("✅ Predict")

# -------------------------
# Predict (no crash + clear errors)
# -------------------------
if submitted:
    errs = validate_inputs(user_input)
    if errs:
        st.error("Please fix the following before predicting:")
        for e in errs:
            st.write(f"- {e}")
        st.stop()

    try:
        pred = model.predict(user_input)[0]

        proba = None
        classes = None
        pos_idx = None

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(user_input)[0]
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                if 1 in classes:
                    pos_idx = classes.index(1)
                else:
                    pos_idx = 1 if len(classes) > 1 else 0
            else:
                classes = [f"class_{i}" for i in range(len(proba))]
                pos_idx = 1 if len(proba) > 1 else 0

        st.divider()
        st.subheader("📌 Results")

        st.success(f"Prediction: **{pred}**")

        if proba is not None and pos_idx is not None:
            risk = float(proba[pos_idx])
            st.metric("Estimated risk (positive class probability)", f"{risk:.3f}")
            st.progress(min(max(risk, 0.0), 1.0))

            if risk >= 0.70:
                st.warning("High estimated risk. Consider follow-up screening and professional advice.")
            elif risk >= 0.40:
                st.info("Moderate estimated risk. Regular check-ups and healthier habits may help reduce risk.")
            else:
                st.success("Lower estimated risk based on the model inputs.")

            st.markdown("**Prediction probabilities:**")
            st.dataframe(pd.DataFrame([proba], columns=classes), use_container_width=True)

        with st.expander("Show full model input (debug)"):
            st.dataframe(user_input, use_container_width=True)

    except Exception as e:
        st.error(
            "Prediction failed — usually caused by **feature mismatch**.\n\n"
            "✅ Fix: Re-export your model with embedded feature names and ensure Streamlit builds the same columns."
        )
        st.exception(e)

st.caption("Model: cardiovascular_health_model.pkl • Input validation + user-facing errors • Smooth demo-ready UI")