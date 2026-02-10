import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import sklearn

"""
Streamlit application for screening cardiovascular health risk.

This app loads a pre–trained machine‑learning model from
``cardiovascular_health_model.pkl`` and presents a clean, interactive
interface for users to input their health and lifestyle details.  It
validates input ranges, encodes categorical variables to match the
training pipeline, and computes an estimated probability of higher
cardiovascular risk.

If the model file cannot be loaded (for example due to version
incompatibilities), the app still runs and clearly explains the
situation to the user instead of crashing.
"""

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="Cardiovascular Health Risk Screener",
    page_icon="❤️",
    layout="wide",
)

MODEL_PATH = "cardiovascular_health_model.pkl"

# ------------------------------------------------------------------------------
# Category definitions and expected one‑hot columns
# ------------------------------------------------------------------------------

# All categories in the original training data are listed here.  Ordering is
# important because pd.get_dummies(drop_first=True) will drop the first entry in
# each list when encoding.  If you retrain the model with different category
# orders, make sure to update this mapping accordingly.
CATS: dict[str, list[str]] = {
    "General_Health": ["Excellent", "Very Good", "Good", "Fair", "Poor"],
    "Checkup": [
        "Within the past year",
        "Within the past 2 years",
        "Within the past 5 years",
        "5 or more years ago",
        "Never",
    ],
    "Exercise": ["No", "Yes"],
    "Skin_Cancer": ["No", "Yes"],
    "Other_Cancer": ["No", "Yes"],
    "Depression": ["No", "Yes"],
    "Diabetes": ["No", "Yes"],
    "Arthritis": ["No", "Yes"],
    "Sex": ["Female", "Male"],
    "Age_Category": [
        "18-24",
        "25-29",
        "30-34",
        "35-39",
        "40-44",
        "45-49",
        "50-54",
        "55-59",
        "60-64",
        "65-69",
        "70-74",
        "75-79",
        "80+",
    ],
    "Smoking_History": ["No", "Yes"],
}

# Numeric feature names as used in the training data
NUM_COLS: list[str] = [
    "Height_(cm)",
    "Weight_(kg)",
    "BMI",
    "Alcohol_Consumption",
    "Fruit_Consumption",
    "Green_Vegetables_Consumption",
    "FriedPotato_Consumption",
]

# Categorical feature names
CAT_COLS: list[str] = list(CATS.keys())

def expected_dummy_columns() -> list[str]:
    """
    Assemble the list of one‑hot encoded column names expected by the model.

    When training, pd.get_dummies(X, drop_first=True) was used, so the first
    category for each variable is omitted.  This function reconstructs the full
    column order used at inference time.
    """
    cols: list[str] = []
    cols.extend(NUM_COLS)
    for col in CAT_COLS:
        # Skip the first entry for drop_first=True
        for cat in CATS[col][1:]:
            cols.append(f"{col}_{cat}")
    return cols

EXPECTED_COLS: list[str] = expected_dummy_columns()

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def safe_float(x: str | float | None) -> float | None:
    """Convert a value to float if possible; return None on failure."""
    try:
        return float(x)  # type: ignore[return-value]
    except Exception:
        return None

def show_env_versions() -> None:
    """
    Display the runtime versions of Python and key libraries.  This helps users
    and graders diagnose issues when model loading fails due to mismatched
    library versions.
    """
    st.caption(
        f"Runtime versions — "
        f"Python: {sys.version.split()[0]} | "
        f"NumPy: {np.__version__} | "
        f"pandas: {pd.__version__} | "
        f"scikit‑learn: {sklearn.__version__} | "
        f"joblib: {joblib.__version__}"
    )

@st.cache_resource(show_spinner=False)
def try_load_model(path: str) -> tuple[object | None, str | None]:
    """
    Attempt to load the pickled model from the given path.

    Returns a tuple (model, error_message).  Exactly one of model or
    error_message will be non‑None.  The function never calls st.stop() so
    that the app remains operational even when the model is missing or
    incompatible.
    """
    if not os.path.exists(path):
        return None, f"Model file not found: {path}"
    try:
        model = joblib.load(path)
        return model, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

def validate_inputs(
    h_cm: float | None,
    w_kg: float | None,
    bmi: float | None,
    alcohol: float | None,
    fruit: float | None,
    veg: float | None,
    fried: float | None,
) -> list[str]:
    """
    Validate user inputs against sensible physiological and dietary ranges.
    Returns a list of error messages; an empty list indicates all inputs look
    acceptable.
    """
    errs: list[str] = []

    # Height
    if h_cm is None or h_cm <= 0:
        errs.append("Height must be a positive number.")
    elif h_cm < 100 or h_cm > 230:
        errs.append("Height looks unusual. Enter a value between 100 and 230 cm.")

    # Weight
    if w_kg is None or w_kg <= 0:
        errs.append("Weight must be a positive number.")
    elif w_kg < 25 or w_kg > 300:
        errs.append("Weight looks unusual. Enter a value between 25 and 300 kg.")

    # BMI
    if bmi is None or bmi <= 0:
        errs.append("BMI must be a positive number.")
    elif bmi < 10 or bmi > 60:
        errs.append("BMI looks unusual. Enter a value between 10 and 60.")

    # Consumption fields
    for name, val, hi in [
        ("Alcohol consumption", alcohol, 200),
        ("Fruit consumption", fruit, 50),
        ("Green vegetables consumption", veg, 50),
        ("Fried potato consumption", fried, 50),
    ]:
        if val is None or val < 0:
            errs.append(f"{name} must be zero or more.")
        elif val > hi:
            errs.append(f"{name} looks too high. Use ≤ {hi}.")

    return errs

def encode_like_training(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features exactly as during model training.

    The function sets category types with the original order (so that
    pd.get_dummies drop_first behaves the same), performs one‑hot encoding, and
    aligns columns with EXPECTED_COLS (missing columns are filled with zeros).
    """
    for col in CAT_COLS:
        raw_df[col] = pd.Categorical(raw_df[col], categories=CATS[col], ordered=True)

    encoded = pd.get_dummies(raw_df, drop_first=True)
    aligned = encoded.reindex(columns=EXPECTED_COLS, fill_value=0)
    return aligned.astype(float)

def predict_proba_1(model: object, X: pd.DataFrame) -> float:
    """
    Predict the probability of the positive class (higher risk) given model and input.

    If the model has predict_proba, use that; otherwise, fall back to a hard
    classification prediction (0 or 1).
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # handle multi‑class gracefully (should not happen here)
        if isinstance(proba, np.ndarray) and proba.shape[1] >= 2:
            return float(proba[0, 1])
        return float(proba[0])
    # Hard fallback
    pred = model.predict(X)[0]
    return 1.0 if int(pred) == 1 else 0.0

# ------------------------------------------------------------------------------
# Application layout
# ------------------------------------------------------------------------------

st.title("❤️ Cardiovascular Health Risk Screener")
st.caption("Risk screening only — not a medical diagnosis.")
show_env_versions()

model, model_err = try_load_model(MODEL_PATH)

with st.expander("Important note", expanded=True):
    st.write(
        (
            "This app estimates cardiovascular risk using a trained machine‑learning model.\n\n"
            "It is **not** a clinical diagnosis.  If you are worried about your health, "
            "please consult a licensed healthcare professional."
        )
    )

if model is None:
    st.error("The risk model could not be loaded, so predictions are currently unavailable.")
    st.warning(
        (
            "This typically indicates that the model file is missing or was saved in an "
            "environment with different versions of Python, NumPy or scikit‑learn.\n\n"
            "If you control the deployment environment, pin the Python version to 3.10 or 3.11 "
            "and ensure that NumPy and scikit‑learn versions match those used when training "
            "the model.  Otherwise, re‑save the model in the same runtime where the app is "
            "deployed."
        )
    )
    if model_err:
        with st.expander("Technical details"):
            st.code(model_err, language="text")

#
# Input form on the left
#
left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    st.subheader("1. Enter your details")

    # Define options for select boxes
    GENERAL_HEALTH_UI = ["Excellent", "Very Good", "Good", "Fair", "Poor"]
    CHECKUP_UI = [
        "Within the past year",
        "Within the past 2 years",
        "Within the past 5 years",
        "5 or more years ago",
        "Never",
    ]
    YES_NO = ["No", "Yes"]
    SEX_UI = ["Female", "Male"]
    AGE_UI = CATS["Age_Category"]

    with st.form(key="risk_form", clear_on_submit=False):
        # Demographics and medical history
        c1, c2 = st.columns(2)
        with c1:
            general_health = st.selectbox("General health", GENERAL_HEALTH_UI, index=2)
            checkup = st.selectbox("Last medical check‑up", CHECKUP_UI, index=0)
            age_category = st.selectbox("Age group", AGE_UI, index=0)
            sex = st.selectbox("Sex", SEX_UI, index=0)
        with c2:
            exercise = st.selectbox("Exercise", YES_NO, index=1)
            smoking_history = st.selectbox("Smoking history", YES_NO, index=0)
            diabetes = st.selectbox("Diabetes", YES_NO, index=0)
            depression = st.selectbox("Depression", YES_NO, index=0)
        # Additional conditions
        c3, c4 = st.columns(2)
        with c3:
            skin_cancer = st.selectbox("Skin cancer", YES_NO, index=0)
            other_cancer = st.selectbox("Other cancer", YES_NO, index=0)
        with c4:
            arthritis = st.selectbox("Arthritis", YES_NO, index=0)
            # Empty placeholder to align layout
            st.write("")
        st.divider()

        # Body measurements
        st.subheader("Body measurements")
        bm1, bm2, bm3 = st.columns(3)
        with bm1:
            height_cm = st.number_input(
                "Height (cm)",
                min_value=100.0,
                max_value=230.0,
                value=170.0,
                step=0.1,
                format="%.1f",
                help="Enter your height in centimeters.",
            )
        with bm2:
            weight_kg = st.number_input(
                "Weight (kg)",
                min_value=25.0,
                max_value=300.0,
                value=70.0,
                step=0.1,
                format="%.1f",
                help="Enter your weight in kilograms.",
            )
        with bm3:
            bmi_mode = st.radio(
                "BMI",
                options=["Auto‑calculate", "Enter manually"],
                horizontal=True,
                index=0,
            )
            if bmi_mode == "Auto‑calculate":
                # Compute BMI when height and weight are valid
                if height_cm > 0:
                    bmi_val = weight_kg / ((height_cm / 100.0) ** 2)
                    st.text_input("BMI (auto)", value=f"{bmi_val:.2f}", disabled=True)
                    bmi = bmi_val
                else:
                    st.text_input("BMI (auto)", value="N/A", disabled=True)
                    bmi = None
            else:
                bmi = st.number_input(
                    "BMI",
                    min_value=10.0,
                    max_value=60.0,
                    value=24.2,
                    step=0.1,
                    format="%.1f",
                    help="Enter your Body Mass Index if known.",
                )

        st.divider()

        # Diet and lifestyle
        st.subheader("Diet & lifestyle (weekly frequency)")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            alcohol = st.number_input(
                "Alcohol (units)",
                min_value=0.0,
                max_value=200.0,
                value=0.0,
                step=1.0,
                format="%.0f",
                help="Approximate units of alcohol consumed per week.",
            )
        with d2:
            fruit = st.number_input(
                "Fruit (servings)",
                min_value=0.0,
                max_value=50.0,
                value=1.0,
                step=1.0,
                format="%.0f",
                help="Number of fruit servings you eat per week.",
            )
        with d3:
            veg = st.number_input(
                "Green veg (servings)",
                min_value=0.0,
                max_value=50.0,
                value=1.0,
                step=1.0,
                format="%.0f",
                help="Number of green vegetable servings per week.",
            )
        with d4:
            fried = st.number_input(
                "Fried potatoes (servings)",
                min_value=0.0,
                max_value=50.0,
                value=0.0,
                step=1.0,
                format="%.0f",
                help="Number of fried potato servings (chips/fries) per week.",
            )

        # Submit button
        submitted = st.form_submit_button("Predict risk")

#
# Results on the right
#
with right_col:
    st.subheader("2. Results")
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    # Process submission
    if submitted:
        errors = validate_inputs(height_cm, weight_kg, bmi, alcohol, fruit, veg, fried)
        if errors:
            st.error("Please correct the following:")
            for msg in errors:
                st.write(f"• {msg}")
        elif model is None:
            st.error("Predictions are unavailable because the model could not be loaded.")
        else:
            # Construct a single‑row DataFrame with original column names
            raw_df = pd.DataFrame(
                [
                    {
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
                        "BMI": float(bmi) if bmi is not None else np.nan,
                        "Alcohol_Consumption": float(alcohol),
                        "Fruit_Consumption": float(fruit),
                        "Green_Vegetables_Consumption": float(veg),
                        "FriedPotato_Consumption": float(fried),
                    }
                ]
            )
            try:
                X_input = encode_like_training(raw_df)
                probability = predict_proba_1(model, X_input)
                risk_label = "Higher risk" if probability >= 0.5 else "Lower risk"
                st.session_state.last_result = {
                    "prob": probability,
                    "label": risk_label,
                    "inputs": X_input,
                }
            except Exception as e:
                st.error("Prediction failed due to a data processing error.")
                st.exception(e)

    # Display results if available
    result = st.session_state.get("last_result")
    if result is None:
        st.info("Fill in the form on the left and click **Predict risk** to see your result.")
    else:
        prob_val = result["prob"]
        risk_label = result["label"]

        st.metric("Risk level", risk_label)
        st.write(f"Estimated probability of higher cardiovascular risk: **{prob_val*100:.1f}%**")
        # Show progress bar representing probability
        st.progress(min(max(prob_val, 0.0), 1.0))

        st.divider()
        st.subheader("Recommendations")
        st.write(
            (
                "These general recommendations may help support cardiovascular health:\n\n"
                "- Schedule regular check‑ups with your healthcare provider.\n"
                "- Stay physically active and incorporate both aerobic and strength exercises.\n"
                "- Eat a balanced diet rich in fruits, vegetables, whole grains and lean proteins.\n"
                "- Get adequate sleep and manage stress levels.\n"
                "- Seek medical advice if you experience concerning symptoms or have existing conditions."
            )
        )

        with st.expander("Show encoded model input (advanced)"):
            st.dataframe(result["inputs"], use_container_width=True)

# End of file