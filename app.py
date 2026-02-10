# app.py
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Cardiovascular Health Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Model + schema (must match training)
# -------------------------
MODEL_FILENAME = "cardiovascular_health_model.pkl"
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

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
# Styling (clean, modern, rubric-friendly)
# -------------------------
st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; max-width: 1280px; }

.card {
  border: 1px solid rgba(120,120,120,0.18);
  border-radius: 18px;
  padding: 18px 18px 14px 18px;
  background: rgba(255,255,255,0.02);
  box-shadow: 0 8px 18px rgba(0,0,0,0.06);
}
.card h3 { margin-top: 0; margin-bottom: 0.35rem; }
.muted { color: rgba(140,140,140,0.95); font-size: 0.95rem; }

label, .stMarkdown p { font-size: 0.98rem; }

.stButton>button, .stForm button {
  border-radius: 12px !important;
  padding: 0.55rem 0.9rem !important;
}

.sidebar-title { font-size: 1.05rem; font-weight: 700; margin-bottom: 0.4rem; }

.badge {
  display:inline-block; padding: 0.22rem 0.55rem; border-radius: 999px;
  border: 1px solid rgba(120,120,120,0.22);
  font-weight: 700; font-size: 0.9rem;
}
.badge-low { background: rgba(34,197,94,0.12); }
.badge-mid { background: rgba(234,179,8,0.14); }
.badge-high{ background: rgba(239,68,68,0.14); }

.hr { height:1px; background: rgba(140,140,140,0.20); margin: 0.8rem 0 0.8rem 0; }

a[data-testid="stHeaderActionElements"] { display:none; }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Helpers
# -------------------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        return None, f"Model file not found: {MODEL_FILENAME} (place it next to app.py)."
    try:
        return joblib.load(path), None
    except Exception:
        return None, "Model file exists but could not be loaded in this environment (pickle/dependency mismatch)."


def validate_inputs(h_cm, w_kg, bmi, alcohol, fruit, veg, fried):
    errors = []

    # Body measurements (no max limits, but cannot be negative/zero)
    if h_cm is None or h_cm <= 0:
        errors.append("Height must be a positive number.")

    if w_kg is None or w_kg <= 0:
        errors.append("Weight must be a positive number.")

    if bmi is None or bmi <= 0:
        errors.append("BMI could not be calculated. Check height/weight.")

    # Monthly consumption (cannot be negative; keep reasonable upper bounds)
    bounds = [
        ("Alcohol (drinks/month)", alcohol, 200),
        ("Fruit (servings/month)", fruit, 200),
        ("Green vegetables (servings/month)", veg, 200),
        ("Fried potatoes (servings/month)", fried, 200),
    ]
    for name, val, hi in bounds:
        if val is None or val < 0:
            errors.append(f"{name} must be 0 or more.")
        elif val > hi:
            errors.append(f"{name} looks too high. Use ≤ {hi}.")
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


def risk_bucket(prob: float):
    if prob < 0.33:
        return "Lower risk", "badge badge-low"
    if prob < 0.66:
        return "Moderate risk", "badge badge-mid"
    return "Higher risk", "badge badge-high"


def pretty_pct(prob: float) -> str:
    return f"{prob * 100:.1f}%"


# -------------------------
# Header (kept, but removed the extra "Important" card)
# -------------------------
st.markdown(
    """
<div class="card">
  <h2 style="margin:0;">❤️ Cardiovascular Health Risk Screener</h2>
  <div class="muted">
    A quick ML-based screening estimate using your inputs. <b>Not a diagnosis</b>.
    Monthly lifestyle inputs are interpreted <b>per month</b>.
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -------------------------
# Load model (user-facing errors)
# -------------------------
model, model_err = load_model(MODEL_PATH)
if model is None:
    st.error("Prediction is unavailable right now.")
    st.info(model_err)
    st.stop()

# -------------------------
# Session state
# -------------------------
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">Controls</div>', unsafe_allow_html=True)
    show_debug = st.toggle("Show debug (model input)", value=False)
    compact_tips = st.toggle("Compact tips", value=True)
    if st.button("Reset results", use_container_width=True):
        st.session_state.last_result = None
        st.session_state.last_inputs = None
        st.rerun()

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.caption("Model file must be next to app.py:")
    st.code(MODEL_FILENAME, language="text")

# -------------------------
# Main layout
# -------------------------
left, right = st.columns([1.05, 0.95], gap="large")

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

with left:
    # Removed the outer "card" box above Enter details (as requested)
    st.subheader("Enter details")

    tab_health, tab_body, tab_life = st.tabs(["Health", "Body", "Lifestyle (Monthly)"])

    with st.form("risk_form", clear_on_submit=False):
        with tab_health:
            c1, c2 = st.columns(2)

            with c1:
                general_health = st.selectbox(
                    "General health",
                    GENERAL_HEALTH_UI,
                    index=2,
                    help="Your overall self-rated health (how you feel in general).",
                )
                checkup = st.selectbox(
                    "Last medical checkup",
                    CHECKUP_UI,
                    index=0,
                    help="When you last had a routine medical checkup.",
                )
                age_category = st.selectbox(
                    "Age group",
                    AGE_UI,
                    index=0,
                    help="Select the age range that matches you.",
                )
                sex = st.selectbox(
                    "Sex",
                    SEX_UI,
                    index=0,
                    help="Used as an input feature for the model.",
                )

            with c2:
                exercise = st.radio(
                    "Do you exercise?",
                    YES_NO,
                    index=1,
                    horizontal=True,
                    help="Choose Yes if you do any regular physical activity/exercise.",
                )
                smoking_history = st.radio(
                    "Smoking history",
                    YES_NO,
                    index=0,
                    horizontal=True,
                    help="Choose Yes if you have a smoking history.",
                )
                diabetes = st.radio(
                    "Diabetes",
                    YES_NO,
                    index=0,
                    horizontal=True,
                    help="Choose Yes if you have been told you have diabetes.",
                )
                depression = st.radio(
                    "Depression",
                    YES_NO,
                    index=0,
                    horizontal=True,
                    help="Choose Yes if you have been told you have depression.",
                )

            c3, c4 = st.columns(2)
            with c3:
                skin_cancer = st.radio(
                    "Skin cancer",
                    YES_NO,
                    index=0,
                    horizontal=True,
                    help="Choose Yes if you have been told you have skin cancer.",
                )
                other_cancer = st.radio(
                    "Other cancer",
                    YES_NO,
                    index=0,
                    horizontal=True,
                    help="Choose Yes if you have been told you have any other cancer.",
                )
            with c4:
                arthritis = st.radio(
                    "Arthritis",
                    YES_NO,
                    index=0,
                    horizontal=True,
                    help="Choose Yes if you have been told you have arthritis.",
                )

        with tab_body:
            st.caption("BMI is auto-calculated from height and weight.")
            m1, m2, m3 = st.columns([0.34, 0.34, 0.32])

            with m1:
                height_cm = st.number_input(
                    "Height (cm)",
                    min_value=0.0,
                    value=170.0,
                    step=1.0,
                    help="Enter your height in centimetres (e.g., 170). Must be > 0.",
                )
            with m2:
                weight_kg = st.number_input(
                    "Weight (kg)",
                    min_value=0.0,
                    value=70.0,
                    step=1.0,
                    help="Enter your weight in kilograms (e.g., 70). Must be > 0.",
                )

            bmi = None
            if height_cm and height_cm > 0:
                bmi = float(weight_kg) / ((float(height_cm) / 100.0) ** 2)

            with m3:
                st.metric("BMI (auto)", f"{bmi:.2f}" if bmi is not None else "—")

            if bmi is not None:
                if bmi < 18.5:
                    bmi_note = "Below typical range"
                elif bmi < 25:
                    bmi_note = "Typical range"
                elif bmi < 30:
                    bmi_note = "Above typical range"
                else:
                    bmi_note = "High range"
                st.caption(f"BMI note: **{bmi_note}** (general guide only).")

        with tab_life:
            st.caption("All values below are interpreted **per month**.")
            d1, d2 = st.columns(2)
            with d1:
                alcohol = st.slider(
                    "Alcohol drink consumption (drinks/month)",
                    min_value=0,
                    max_value=200,
                    value=0,
                    help="Total number of alcoholic drinks in an average month.",
                )
                fried = st.slider(
                    "Fried potato consumption (servings/month)",
                    min_value=0,
                    max_value=200,
                    value=0,
                    help="Total servings in an average month (e.g., fries/chips).",
                )
            with d2:
                fruit = st.slider(
                    "Fruit consumption (servings/month)",
                    min_value=0,
                    max_value=200,
                    value=30,
                    help="Total fruit servings in an average month (rough estimate is fine).",
                )
                veg = st.slider(
                    "Green vegetables consumption (servings/month)",
                    min_value=0,
                    max_value=200,
                    value=30,
                    help="Total green-veg servings in an average month (rough estimate is fine).",
                )

            if not compact_tips:
                st.info(
                    "If you’re unsure, estimate using weekly habits × 4.\n"
                    "- Example: 3 fruit servings/week → ~12/month."
                )

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict risk", use_container_width=True)

with right:
    # Removed the outer "card" box above Results (as requested)
    st.subheader("Results")

    if st.session_state.last_result is None:
        st.info("Complete the form and click **Predict risk** to see your result.")
    else:
        prob = st.session_state.last_result["prob"]
        bucket, badge_class = risk_bucket(prob)

        st.markdown(
            f"""
<div style="display:flex; align-items:center; justify-content:space-between; gap:14px;">
  <div>
    <div class="muted" style="margin-bottom:6px;">Risk level</div>
    <div class="{badge_class}">{bucket}</div>
  </div>
  <div style="text-align:right;">
    <div class="muted" style="margin-bottom:6px;">Estimated probability</div>
    <div style="font-size:1.6rem; font-weight:800;">{pretty_pct(prob)}</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.progress(min(max(prob, 0.0), 1.0))
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        st.markdown("#### What this means")
        st.write(
            "- This is a **screening estimate** based on your inputs and the trained model.\n"
            "- A higher estimate does **not** confirm a condition, and a lower estimate does **not** guarantee no risk.\n"
            "- If you’re concerned, a proper checkup is the best next step."
        )

        st.markdown("#### Next steps (general)")
        st.write(
            "- Consider a health checkup if you have concerns.\n"
            "- Maintain healthy habits: regular activity, balanced diet, enough sleep.\n"
            "- If you feel unwell or have symptoms, seek medical advice."
        )

        if show_debug:
            with st.expander("Debug: Encoded model input (X)"):
                st.dataframe(st.session_state.last_result["X"], use_container_width=True)

# -------------------------
# Prediction action
# -------------------------
if submitted:
    try:
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
            "BMI": float(bmi) if bmi is not None else None,
            "Alcohol_Consumption": float(alcohol),
            "Fruit_Consumption": float(fruit),
            "Green_Vegetables_Consumption": float(veg),
            "FriedPotato_Consumption": float(fried),
        }])

        errs = validate_inputs(
            safe_float(height_cm),
            safe_float(weight_kg),
            safe_float(bmi) if bmi is not None else None,
            safe_float(alcohol),
            safe_float(fruit),
            safe_float(veg),
            safe_float(fried),
        )

        if errs:
            st.session_state.last_result = None
            st.session_state.last_inputs = None
            st.error("Please fix these before predicting:")
            for e in errs:
                st.write(f"- {e}")
        else:
            X_input = encode_like_training(raw_df)
            prob = predict_proba_1(model, X_input)

            st.session_state.last_result = {"prob": float(prob), "X": X_input}
            st.session_state.last_inputs = raw_df.to_dict(orient="records")[0]
            st.success("Prediction updated.")
            st.rerun()

    except Exception:
        st.session_state.last_result = None
        st.error("Prediction failed. This usually means a model/features mismatch.")

# -------------------------
# Footer
# -------------------------
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.caption("")