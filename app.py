import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "model.joblib"
DATA_PATH = "dataset.csv"


@st.cache_resource
def load_model():
    """Load the trained model if available, otherwise train a quick fallback model."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(DATA_PATH)
        feature_cols = [
            "age",
            "sex",
            "chest pain type",
            "resting bp s",
            "cholesterol",
            "fasting blood sugar",
            "resting ecg",
            "max heart rate",
            "exercise angina",
            "oldpeak",
            "ST slope",
        ]
        X = df[feature_cols]
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            max_depth=None,
        )
        model.fit(X_train, y_train)

        joblib.dump(model, MODEL_PATH)
        return model


st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="centered")
st.title("Heart Disease Prediction App")
st.caption("Predict whether a patient is likely to have heart disease using the trained ML model.")

feature_cols = [
    "age",
    "sex",
    "chest pain type",
    "resting bp s",
    "cholesterol",
    "fasting blood sugar",
    "resting ecg",
    "max heart rate",
    "exercise angina",
    "oldpeak",
    "ST slope",
]

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 90, 50)
        sex = st.selectbox("Sex", [0, 1], index=1)
        chest_pain_type = st.selectbox("Chest pain type", [1, 2, 3, 4], index=2)
        resting_bp_s = st.slider("Resting blood pressure (mmHg)", 80, 200, 120)
        cholesterol = st.slider("Cholesterol", 0, 600, 200)
        fasting_blood_sugar = st.selectbox("Fasting blood sugar", [0, 1], index=0)
        resting_ecg = st.selectbox("Resting ECG", [0, 1, 2], index=0)

    with col2:
        max_heart_rate = st.slider("Max heart rate", 60, 202, 140)
        exercise_angina = st.selectbox("Exercise-induced angina", [0, 1], index=0)
        oldpeak = st.slider("Oldpeak", 0.0, 6.5, 1.0, step=0.1)
        st_slope = st.selectbox("ST slope", [1, 2, 3], index=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    model = load_model()
    input_df = pd.DataFrame(
        [{
            "age": age,
            "sex": sex,
            "chest pain type": chest_pain_type,
            "resting bp s": resting_bp_s,
            "cholesterol": cholesterol,
            "fasting blood sugar": fasting_blood_sugar,
            "resting ecg": resting_ecg,
            "max heart rate": max_heart_rate,
            "exercise angina": exercise_angina,
            "oldpeak": oldpeak,
            "ST slope": st_slope,
        }]
    )

    raw_prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0]

    if raw_prediction[0] == 1:
        label = "High risk of heart disease"
        color = "#F63366"
    else:
        label = "Low risk of heart disease"
        color = "#00B894"

    st.markdown(
        f"<div style='padding:20px; border-radius:10px; background:{color}; color:white; font-size:24px; font-weight:600;'>"
        f"Prediction: {label}</div>",
        unsafe_allow_html=True,
    )

    st.write(f"Probability of disease: {probability[1] * 100:.2f}%")
    st.write(f"Probability of no disease: {probability[0] * 100:.2f}%")

st.sidebar.header("Model info")
st.sidebar.write("This app uses a Random Forest classifier trained on the repo's dataset.csv file.")
st.sidebar.write("It predicts whether a patient is likely to have heart disease.")
