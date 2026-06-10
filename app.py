import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------
# PAGE SETTINGS
# -----------------------

st.set_page_config(
    page_title="Loan Approval Dashboard",
    page_icon="🏦",
    layout="wide"
)

# -----------------------
# LOAD FILES
# -----------------------

base_dir = Path(__file__).resolve().parent

df = pd.read_csv(
    base_dir / "dataset.csv"
)

model = joblib.load(
    base_dir / "best_model.pkl"
)

# -----------------------
# HEADER
# -----------------------

st.title("🏦 Loan Approval Prediction System")

st.markdown(
    """
    Predict whether a loan application
    is approved or rejected using
    Machine Learning.
    """
)

# -----------------------
# DATASET SECTION
# -----------------------

st.subheader("Dataset Preview")

st.dataframe(df)

st.write("Dataset Shape:", df.shape)

# -----------------------
# SIDEBAR INPUTS
# -----------------------

st.sidebar.header("Applicant Details")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

married = st.sidebar.selectbox(
    "Married",
    ["Yes", "No"]
)

income = st.sidebar.number_input(
    "Income",
    min_value=0,
    value=5000
)

loan_amount = st.sidebar.number_input(
    "Loan Amount",
    min_value=0,
    value=150
)

education = st.sidebar.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

# -----------------------
# ENCODE USER INPUT
# -----------------------

gender_value = 1 if gender == "Male" else 0

married_value = 1 if married == "Yes" else 0

education_value = 0 if education == "Graduate" else 1

# -----------------------
# PREDICTION BUTTON
# -----------------------

if st.sidebar.button("Predict Loan Status"):

    input_data = pd.DataFrame(
        [[
            gender_value,
            married_value,
            income,
            loan_amount,
            education_value
        ]],
        columns=[
            "Gender",
            "Married",
            "Income",
            "LoanAmount",
            "Education"
        ]
    )

    prediction = model.predict(
        input_data
    )

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.success(
            "✅ Loan Approved"
        )

    else:

        st.error(
            "❌ Loan Rejected"
        )

# -----------------------
# DISPLAY PROJECT GRAPHS
# -----------------------

st.subheader("Model Evaluation")

try:

    st.image(
        str(base_dir / "confusion_matrix.png"),
        caption="Confusion Matrix"
    )

    st.image(
        str(base_dir / "roc_curve.png"),
        caption="ROC Curve"
    )

    st.image(
        str(base_dir / "feature_importance.png"),
        caption="Feature Importance"
    )

    st.image(
        str(base_dir / "accuracy_comparison.png"),
        caption="Model Comparison"
    )

except:
    st.warning(
        "Run project.py first to generate graphs."
    )

# -----------------------
# FOOTER
# -----------------------

st.markdown("---")

st.markdown(
    "Developed using Python, Streamlit, Pandas, Scikit-Learn and Machine Learning."
)