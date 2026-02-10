# app.py
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# CONFIG
# =========================================================
APP_TITLE = "❤️ Cardiovascular Health Risk Screener"
MODEL_PATH = "cardiovascular_health_model.pkl"
DATA_PATH = "CVD_cleaned.csv"          # used to rebuild exact dummy columns (same as training)
TARGET_COL = "Heart_Disease"           # must match your notebook

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(page_title=APP_TITLE, page_icon="❤️", layout="centered")
st.title(APP_TITLE)
st.caption("Risk screening tool (not a medical diagnosis). For educational demo use.")

st.info(
    "This app estimates **cardiovascular risk** using your trained ML model.\n\n"
    "- **Not a medical diagnosis**\n"
    "- For real concerns, consult a qualified healthcare professional."
)

# =========================================================
# LOAD MODEL + BUILD FEATURE TEMPLATE FROM TRAINING DATA
# =========================================================
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

@st.cache_data
def load_training_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data
def build_template_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str], dict[str, list[str]]]:
    """
    Rebuild the EXACT one-hot encoded feature columns using the same approach used in your notebook:
      X = df.drop(["Heart_Disease"])
      X = pd.get_dummies(X, drop_first=True)

    Returns:
      feature_columns: columns expected by the model
      numeric_cols: raw numeric input cols (before dummies)
      categorical_cols: raw categorical input cols (before dummies)
      cat_options: mapping of categorical col -> sorted unique options (strings)
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Dataset is missing target column: {TARGET_COL}")

    raw_X = df.drop(columns=[TARGET_COL]).copy()

    # Identify raw numeric vs categorical
    numeric_cols = raw_X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in raw_X.columns if c not in numeric_cols]

    # Build options for categorical fields (sorted, stable)
    cat_options = {}
    for c in categorical_cols:
        # keep as string to avoid Streamlit weirdness with mixed types
        opts = sorted(raw_X[c].dropna().astype(str).unique().tolist())
        cat_options[c] = opts

    # Build template dummy columns exactly as training
    X_template = pd.get_dummies(raw_X, drop_first=True)
    feature_columns = X_template.columns.tolist()

    return feature_columns, numeric_cols, categorical_cols, cat_options

# Guardrails: files exist
if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found: **{MODEL_PATH}**\n\n"
        "✅ Fix: Put `cardiovascular_health_model.pkl` in the same folder as `app.py`."
    )
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error(
        f"Dataset file not found: **{DATA_PATH}**\n\n"
        "This app uses your training CSV to rebuild the exact one-hot columns (because the model does not store feature names).\n"
        "✅ Fix: Put `CVD_cleaned.csv` in the same folder as `app.py` (and include it in GitHub for deployment)."
    )
    st.stop()

# Load resources
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error("Failed to load the model. The .pkl may be corrupted or saved incorrectly.")
    st.exception(e)
    st.stop()

try:
    df_train = load_training_data(DATA_PATH)
    feature_columns, numeric_cols, categorical_cols, cat_options = build_template_columns(df_train)
except Exception as e:
    st.error("Failed to load dataset / build feature template columns.")
    st.exception(e)
    st.stop()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("ℹ️ Quick info")
    st.write(f"**Model:** `{MODEL_PATH}`")
    st.write(f"**Template CSV:** `{DATA_PATH}`")
    st.write(f"**Raw input fields:** {len(numeric_cols) + len(categorical_cols)}")
    st.write(f"**Model features (after one-hot):** {len(feature_columns)}")
    st.divider()
    st.write(
        "If your model does **not** include feature names, this is the safest method:\n"
        "- Use the same CSV structure\n"
        "- Rebuild one-hot columns with `drop_first=True`\n"
        "- Align to the template columns before predicting"
    )

# =========================================================
# INPUT VALIDATION RULES (safe + user-friendly)
# =========================================================
RANGE_HINTS = {
    # Common CVD fields (only applied if those columns exist)
    "BMI": (10.0, 60.0, 25.0),
    "Sleep_Hours": (0.0, 24.0, 7.0),
    "Physical_Health_Days": (0.0, 30.0, 0.0),
    "Mental_Health_Days": (0.0, 30.0, 0.0),
}

def validate_raw_inputs(raw_row: dict) -> list[str]:
    errors = []

    if "BMI" in raw_row:
        try:
            if float(raw_row["BMI"]) <= 0:
                errors.append("BMI must be greater than 0.")
        except Exception:
            errors.append("BMI must be a valid number.")

    if "Sleep_Hours" in raw_row:
        try:
            sh = float(raw_row["Sleep_Hours"])
            if sh < 0 or sh > 24:
                errors.append("Sleep Hours must be between 0 and 24.")
        except Exception:
            errors.append("Sleep Hours must be a valid number.")

    # Generic numeric checks
    for c in numeric_cols:
        v = raw_row.get(c, None)
        if v is None:
            continue
        try:
            fv = float(v)
            if not np.isfinite(fv):
                errors.append(f"{c} is invalid (NaN/Infinity).")
        except Exception:
            errors.append(f"{c} must be a valid number.")

    return errors

# =========================================================
# BUILD UI (highly interactive + responsive)
# =========================================================
st.subheader("🧾 Enter details")

raw_input = {}

with st.form("predict_form", clear_on_submit=False):
    # ---------- Numeric ----------
    if numeric_cols:
        st.markdown("### Numeric fields")
        colA, colB = st.columns(2)
        for i, c in enumerate(numeric_cols):
            lo, hi, default = RANGE_HINTS.get(c, (0.0, 100.0, 0.0))
            label = c.replace("_", " ")

            container = colA if i % 2 == 0 else colB
            with container:
                raw_input[c] = st.number_input(
                    label,
                    min_value=float(lo),
                    max_value=float(hi),
                    value=float(default),
                    step=1.0 if (hi - lo) >= 10 else 0.5,
                    help=f"Expected range: {lo} to {hi}",
                )

    # ---------- Categorical ----------
    if categorical_cols:
        st.markdown("### Categorical fields (dropdowns)")
        for c in categorical_cols:
            opts = cat_options.get(c, [])
            if not opts:
                # fallback
                opts = ["(unknown)"]

            # drop_first=True baseline is the first category in sorted order
            baseline = opts[0]
            baseline_label = f"(baseline: {baseline})"
            display_opts = [baseline_label] + opts[1:]

            choice = st.selectbox(
                c.replace("_", " "),
                options=display_opts,
                index=0,
                help="Baseline represents the dropped first category used in one-hot encoding (drop_first=True).",
            )
            if choice == baseline_label:
                raw_input[c] = baseline
            else:
                raw_input[c] = choice

    submitted = st.form_submit_button("✅ Predict")

# =========================================================
# TRANSFORM + PREDICT (no crash + clear user errors)
# =========================================================
if submitted:
    errs = validate_raw_inputs(raw_input)
    if errs:
        st.error("Please fix the following before predicting:")
        for e in errs:
            st.write(f"- {e}")
        st.stop()

    try:
        # Build 1-row raw DataFrame with the SAME raw columns as training
        raw_df = pd.DataFrame([raw_input])

        # Ensure any missing raw columns (rare) are added with safe defaults
        raw_expected_cols = df_train.drop(columns=[TARGET_COL]).columns.tolist()
        for c in raw_expected_cols:
            if c not in raw_df.columns:
                # numeric -> 0, categorical -> baseline if known else empty string
                if c in numeric_cols:
                    raw_df[c] = 0.0
                else:
                    opts = cat_options.get(c, [""])
                    raw_df[c] = opts[0] if opts else ""

        raw_df = raw_df[raw_expected_cols]  # exact order

        # One-hot encode same as training
        X_live = pd.get_dummies(raw_df, drop_first=True)

        # Align to template columns (critical to avoid mismatch)
        X_live = X_live.reindex(columns=feature_columns, fill_value=0)

        # Final safety: ensure numeric dtype
        X_live = X_live.astype(float)

        # Predict
        pred = model.predict(X_live)[0]

        st.divider()
        st.subheader("📌 Results")
        st.success(f"Prediction: **{pred}**")

        # Probabilities + risk bar
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_live)[0]

            classes = list(model.classes_) if hasattr(model, "classes_") else [f"class_{i}" for i in range(len(proba))]

            # Try to pick positive class intelligently (1 if present else second column)
            if 1 in classes:
                pos_idx = classes.index(1)
            else:
                pos_idx = 1 if len(classes) > 1 else 0

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

        with st.expander("Show full transformed model input (debug)"):
            st.dataframe(X_live, use_container_width=True)

    except Exception as e:
        st.error(
            "Prediction failed.\n\n"
            "Common causes:\n"
            "- Model expects different columns than the training CSV\n"
            "- Training CSV is not the same one used when the model was trained\n\n"
            "✅ Fix: Use the same `CVD_cleaned.csv` you trained on, and redeploy."
        )
        st.exception(e)

st.caption("Uses training CSV to rebuild dummy columns (drop_first=True) • Includes validation + friendly errors • Demo-ready")