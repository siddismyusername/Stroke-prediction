import streamlit as st
import pandas as pd
import pickle

# --- Load Saved Objects ---
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="Stroke Prediction App", page_icon="🧠")

# --- UI Elements ---
st.title('🧠 Stroke Risk Prediction App')
st.write(
    "This app uses a machine learning model to predict your risk of stroke. "
    "Please enter the required information below."
)
st.markdown("---")

# Create columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.slider('Age', 1, 100, 50)
    hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
    heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
    ever_married = st.selectbox('Ever Married', ['Yes', 'No'])

with col2:
    work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
    residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
    glucose_level = st.slider('Average Glucose Level (mg/dL)', 50.0, 300.0, 100.0)
    bmi = st.slider('Body Mass Index (BMI)', 10.0, 60.0, 25.0)
    smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

# --- Prediction Logic ---
if st.button("🩺 Predict Stroke Risk"):
    # Encoding maps
    hypertension_map = {'Yes': 1, 'No': 0}
    heart_disease_map = {'Yes': 1, 'No': 0}
    ever_married_map = {'Yes': 1, 'No': 0}
    residence_type_map = {'Urban': 1, 'Rural': 0}
    smoking_status_map = {'formerly smoked': 1, 'smokes': 1, 'never smoked': 0, 'Unknown': 0}
    work_type_map = {'Never_worked': 0, 'children': 0, 'Govt_job': 1, 'Private': 2, 'Self-employed': 3}
    gender_map = {'Male': 1, 'Female': 0}

    # Input data dictionary with mapped values
    input_data = {
        'gender': gender_map[gender],
        'age': age,
        'hypertension': hypertension_map[hypertension],
        'heart_disease': heart_disease_map[heart_disease],
        'ever_married': ever_married_map[ever_married],
        'work_type': work_type_map[work_type],
        'Residence_type': residence_type_map[residence_type],
        'avg_glucose_level': glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status_map[smoking_status]
    }

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Scale the features
    df_scaled = scaler.transform(df)

    # Make prediction
    prediction = model.predict(df_scaled)
    prediction_proba = model.predict_proba(df_scaled)

    # --- Display Results ---
    st.markdown("---")
    st.subheader("Prediction Result:")

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Stroke ({prediction_proba[0][1]:.0%} probability)")
        st.warning("This result is a prediction and not a medical diagnosis. Please consult a healthcare professional for advice.")
    else:
        st.success(f"✅ Low Risk of Stroke ({prediction_proba[0][1]:.0%} probability)")
        st.info("Remember to maintain a healthy lifestyle. Regular check-ups with a doctor are always recommended.")

st.markdown("---")
st.write("*Disclaimer: This app is for informational purposes only and does not substitute for professional medical advice.*")
