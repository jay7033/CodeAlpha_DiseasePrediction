
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
}
h1, h2, h3 {
    color: #183153;
}
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 12px 25px;
    border: none;
    font-weight: bold;
}
.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("❤️ Heart Disease Prediction System")
st.write("Machine Learning Based Heart Disease Risk Classification")
st.info(
    "Educational demonstration only. This model uses a historical "
    "dataset and is not validated for medical diagnosis or treatment."
)

model_path = "model/heart_disease_model.pkl"
imputer_path = "model/imputer.pkl"

if not os.path.exists(model_path) or not os.path.exists(imputer_path):
    st.error("Model files not found. Run python train_model.py first.")
    st.stop()

model = joblib.load(model_path)
imputer = joblib.load(imputer_path)

st.subheader("Enter Patient Medical Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 100, 50)
    sex = st.selectbox(
        "Sex (dataset coding)",
        ["Female", "Male"]
    )
    cp = st.selectbox(
        "Chest Pain Type",
        [
            "Typical Angina",
            "Atypical Angina",
            "Non-Anginal Pain",
            "Asymptomatic"
        ]
    )
    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)", 80, 250, 120
    )
    chol = st.number_input(
        "Serum Cholesterol (mg/dL)", 100, 600, 200
    )

with col2:
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        ["No", "Yes"]
    )
    restecg = st.selectbox(
        "Resting ECG Result",
        ["Normal", "ST-T Abnormality", "LV Hypertrophy"]
    )
    thalach = st.number_input(
        "Maximum Heart Rate Achieved", 60, 220, 150
    )
    exang = st.selectbox(
        "Exercise Induced Angina",
        ["No", "Yes"]
    )

with col3:
    oldpeak = st.number_input(
        "ST Depression (Oldpeak)", 0.0, 7.0, 1.0, 0.1
    )
    slope = st.selectbox(
        "ST Segment Slope",
        ["Upsloping", "Flat", "Downsloping"]
    )
    ca = st.selectbox(
        "Major Vessels Colored by Fluoroscopy",
        [0, 1, 2, 3]
    )
    thal = st.selectbox(
        "Thalassemia Test Result",
        ["Normal", "Fixed Defect", "Reversible Defect"]
    )

st.divider()

if st.button("Predict Heart Disease", use_container_width=True):

    input_data = pd.DataFrame([{
        "age": age,
        "sex": 0 if sex == "Female" else 1,
        "cp": [
            "Typical Angina",
            "Atypical Angina",
            "Non-Anginal Pain",
            "Asymptomatic"
        ].index(cp) + 1,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 0 if fbs == "No" else 1,
        "restecg": [
            "Normal",
            "ST-T Abnormality",
            "LV Hypertrophy"
        ].index(restecg),
        "thalach": thalach,
        "exang": 0 if exang == "No" else 1,
        "oldpeak": oldpeak,
        "slope": [
            "Upsloping",
            "Flat",
            "Downsloping"
        ].index(slope) + 1,
        "ca": ca,
        "thal": {
            "Normal": 3,
            "Fixed Defect": 6,
            "Reversible Defect": 7
        }[thal]
    }])

    processed_data = imputer.transform(input_data)

    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.warning(
            "Model Classification: Heart Disease Pattern Detected"
        )
    else:
        st.success(
            "Model Classification: No Heart Disease Pattern Detected"
        )

    st.metric(
        "Model's Positive-Class Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

    st.caption(
        "This percentage is an uncalibrated machine-learning model "
        "output, not a patient's clinical disease probability. "
        "A negative classification does not rule out heart disease."
    )

st.divider()

st.write("Developed for CodeAlpha Machine Learning Internship")