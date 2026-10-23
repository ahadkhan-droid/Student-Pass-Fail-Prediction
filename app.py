
import streamlit as st
import pandas as pd
import joblib

# Page setup
st.set_page_config(
    page_title="Student Pass Fail Prediction",
    page_icon="🎓"
)

# Load trained Decision Tree model
saved = joblib.load("student_pass_fail_model.pkl")
model = saved["model"]
features = saved["features"]

st.title("🎓 Student Pass/Fail Prediction")
st.write("Decision Tree Machine Learning Model")

# User Inputs
st.subheader("Enter Student Details")

study_hours = st.number_input(
    "Study Hours",
    min_value=0,
    max_value=24,
    value=5
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=75
)

# Predict Result
if st.button("Predict Result"):

    data = pd.DataFrame(
        [[study_hours, attendance]],
        columns=features
    )

    prediction = model.predict(data)[0]

    if prediction == 1:
        st.success("Prediction: PASS ✅")
    else:
        st.error("Prediction: FAIL ❌")

st.caption("Educational demo trained on synthetic student data.")
