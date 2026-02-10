# streamlit.py
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
    "This app estimates **heart disease risk** based on the trained ML model.\n\n"
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
        "✅ Fix: Put **cardiovascular_health_model.pkl** in the SAME folder as this streamlit.py file."
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

    # If you saved a pipeline with preprocessing, try sklearn feature names
    # (works for some transformers that implement get_feature_names_out)
    try:
        if hasattr(m, "get_feature_names_out"):
            cols = list(m.get_feature_names_out())
            if cols:
                return cols
    except Exception:
        pass

    # If pipeline, try last step / preprocessor step
    try:
        if hasattr(m, "named_steps"):
            for step_name, step in m.named_steps.items():
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
        "✅ Fix (recommended): when exporting your model, attach feature names:\n"
        "model.feature_names_ = list(X_train.columns)  # before joblib.dump\n"
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
    return (
        text.replace("_", " ")
            .replace("  ", " ")
            .strip()
    )

def is_binary_col(col: str) -> bool:
    # common dummy/binary patterns
    return col.endswith("_Yes") or col.endswith("_No") or col.endswith("_Positive") or col.endswith("_True") or col.endswith("_1")

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

def set_one_hot(df: pd.DataFrame, prefix: str, chosen_value: str, group_cols: List[str], baseline_label: str = "(baseline)"):
    # clear group
    for c in group_cols:
        df.at[0, c] = 0

    if chosen_value == baseline_label:
        return

    target = f"{prefix}_{chosen_value}"
    if target in df.columns:
        df.at[0, target] = 1

def get_numeric_cols(columns: List[str], dummy_groups: Dict[str, List[str]]) -> List[str]:
    dummy_cols = set()
    for cols in dummy_groups.values():
        dummy_cols.update(cols)

    # numeric columns are those not in dummy groups and not obvious binary dummies
    numeric = []
    for c in columns:
        if c in dummy_cols:
            continue
        # binary columns (single dummy) should be treated separately (selectbox)
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

def safe_float(x, default=None):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return default

# -------------------------
# Build UI structure
# -------------------------
dummy_groups = infer_dummy_groups(feature_columns)
numeric_cols = get_numeric_cols(feature_columns, dummy_groups)
binary_cols = get_single_dummy_binaries(feature_columns, dummy_groups)

user_input = make_empty_input(feature_columns)

# -------------------------
# Sidebar: quick help
# -------------------------
with st.sidebar:
    st.header("ℹ️ How to use")
    st.write(
        "1) Fill in inputs\n"
        "2) Click **Predict**\n"
        "3) View risk + probability\n\n"
        "Tip: If your model was trained with one-hot encoding (drop_first=True), "
        "each dropdown includes a **(baseline)** option that represents “all zeros” for that category group."
    )
    st.divider()
    st.write("**Model file:** cardiovascular_health_model.pkl")
    st.write(f"**Detected features:** {len(feature_columns)}")

# -------------------------
# Input validation rules (simple + safe defaults)
# -------------------------
# You can add more rules if your dataset has known ranges
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

    # if it looks like a "days" feature, give 0-30
    if "Days" in label and col not in RANGE_HINTS:
        lo, hi, default = 0.0, 30.0, 0.0

    # if it looks like a "Consumption" feature, keep it reasonable
    if "Consumption" in label and col not in RANGE_HINTS:
        lo, hi, default = 0.0, 100.0, 0.0

    # Use number_input (no crashes), clamp later
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
    # show friendly yes/no if column name hints it
    options = [0, 1]
    display = {0: "No (0)", 1: "Yes (1)"}
    choice = st.selectbox(label, options, format_func=lambda x: display[x], index=0)
    set_value(user_input, col, int(choice))

def categorical_widget(prefix: str, group_cols: List[str]):
    # dropdown values are the suffixes
    suffixes = [c.replace(prefix + "_", "") for c in group_cols]
    opts = ["(baseline)"] + suffixes

    chosen = st.selectbox(prettify_label(prefix), opts, index=0)
    set_one_hot(user_input, prefix, chosen, group_cols, baseline_label="(baseline)")

def validate_inputs(df: pd.DataFrame) -> List[str]:
    errors = []

    # Example validation: BMI shouldn't be 0 if present (common user mistake)
    if "BMI" in df.columns:
        bmi = safe_float(df.at[0, "BMI"], default=0.0)
        if bmi <= 0:
            errors.append("BMI must be greater than 0.")

    # Sleep hours reasonable if present
    if "Sleep_Hours" in df.columns:
        sh = safe_float(df.at[0, "Sleep_Hours"], default=0.0)
        if sh < 0 or sh > 24:
            errors.append("Sleep Hours must be between 0 and 24.")

    # General numeric sanity: no NaN/inf
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
        # nicer layout in 2 columns
        cols2 = st.columns(2)
        for i, col in enumerate(numeric_cols):
            with cols2[i % 2]:
                numeric_widget(col)

    if dummy_groups:
        st.markdown("#### Categorical fields (dropdowns)")
        # Show in a stable order
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
    # Input validation (user-facing)
    errs = validate_inputs(user_input)
    if errs:
        st.error("Please fix the following before predicting:")
        for e in errs:
            st.write(f"- {e}")
        st.stop()

    # Predict safely
    try:
        pred = model.predict(user_input)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(user_input)[0]

        # Output formatting (friendly)
        st.divider()
        st.subheader("📌 Results")

        # Determine positive class index (if available)
        pos_idx = None
        classes = None
        if proba is not None:
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                # try find class "1"
                if 1 in classes:
                    pos_idx = classes.index(1)
                else:
                    # fallback: assume second column is positive
                    pos_idx = 1 if len(classes) > 1 else 0
            else:
                classes = [f"class_{i}" for i in range(len(proba))]
                pos_idx = 1 if len(proba) > 1 else 0

        # Show main prediction
        pred_text = str(pred)
        st.success(f"Prediction: **{pred_text}**")

        # If probability exists, show risk meter
        if proba is not None and pos_idx is not None:
            risk = float(proba[pos_idx])
            st.metric("Estimated risk (positive class probability)", f"{risk:.3f}")

            st.progress(min(max(risk, 0.0), 1.0))

            if risk >= 0.70:
                st.warning("High estimated risk. Consider follow-up screening and professional advice.")
            elif risk >= 0.40:
                st.info("Moderate estimated risk. Lifestyle factors and check-ups may help reduce risk.")
            else:
                st.success("Lower estimated risk based on the model inputs.")

            st.markdown("**Prediction probabilities:**")
            st.dataframe(pd.DataFrame([proba], columns=classes), use_container_width=True)

        with st.expander("Show full model input (debug)"):
            st.dataframe(user_input, use_container_width=True)

    except Exception as e:
        st.error(
            "Prediction failed — usually caused by **feature mismatch** (your app input columns don’t match training columns).\n\n"
            "✅ Fix: Re-export the model with embedded feature names and ensure Streamlit builds the same columns."
        )
        st.exception(e)

st.caption("Model loaded from cardiovascular_health_model.pkl • App includes validation + user-facing errors • Built for smooth demo/testing")