import streamlit as st
import pandas as pd
import pickle

# --- Load Saved Objects ---
# It's crucial that these 'model.pkl' and 'scaler.pkl' are the ones saved
# from your model training notebook.
try:
    model = pickle.load(open('model.pkl', 'rb'))
    encoder = pickle.load(open('encoder.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'stroke_prediction_model.pkl' and 'scaler.pkl' are in the same directory.")
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
    # 1. Collect and map inputs
    # The mappings must match the encoding done during training
    hypertension_map = {'Yes': 1, 'No': 0}
    heart_disease_map = {'Yes': 1, 'No': 0}
    

    # Create a dictionary of the inputs
    input_data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension_map[hypertension],
        'heart_disease': heart_disease_map[heart_disease],
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': residence_type,
        'avg_glucose_level': glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }

    # 2. Create a DataFrame in the correct order
    df = pd.DataFrame([input_data])
    
    # OHE Encoding
    ohe_features = ['gender','ever_married','work_type','Residence_type','smoking_status']
    encoded=encoder.transform(df[ohe_features]).toarray()
    encoder_df=pd.DataFrame(encoded,columns=encoder.get_feature_names_out(),index=df.index)
    df = pd.concat([df,encoder_df],axis=1)
    df.drop(ohe_features, axis=1, inplace=True)

    # 3. Scale the features
    
    df_scaled = scaler.transform(df)
    # Note: Ensure that the scaler was fitted on the training data during model training

    # 4. Make a prediction
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