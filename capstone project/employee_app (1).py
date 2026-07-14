import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load the trained Gradient Boosting model, label encoders, and feature columns
model = load('gb_model.joblib')
le_dict = load('le_dict.joblib')
feature_cols = load('feature_cols.joblib')

st.set_page_config(page_title="Employee Retention Prediction", page_icon="👔", layout="centered")
st.title("👔 Employee Retention Prediction App")
st.markdown("Fill in the employee details below to predict whether they are likely to look for a new job.")

st.header("Employee Information")

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("City Code", sorted(le_dict['city'].classes_.tolist()), index=sorted(le_dict['city'].classes_.tolist()).index('city_103') if 'city_103' in le_dict['city'].classes_ else 0)
    city_development_index = st.slider("City Development Index", 0.0, 1.0, 0.92, step=0.01)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    relevent_experience = st.selectbox("Relevant Experience", ["Has relevent experience", "No relevent experience"])
    education_level = st.selectbox("Education Level", ["Graduate", "Masters", "Phd", "High School", "Primary School"])
    major_discipline = st.selectbox("Major Discipline", ["STEM", "Business Degree", "Arts", "Humanities", "No Major", "Other"])

with col2:
    enrolled_university = st.selectbox("Enrolled University", ["no_enrollment", "Full time course", "Part time course"])
    experience = st.selectbox("Experience (years)", ["<1"] + [str(i) for i in range(1, 21)] + [">20"])
    company_size = st.selectbox("Company Size", ["<10", "10/49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"])
    company_type = st.selectbox("Company Type", ["Pvt Ltd", "Funded Startup", "Public Sector", "Early Stage Startup", "NGO", "Other"])
    last_new_job = st.selectbox("Years Since Last Job Change", ["never", "1", "2", "3", "4", ">4"])
    training_hours = st.number_input("Training Hours", min_value=0, max_value=400, value=50)

st.markdown("---")

if st.button("🔍 Predict", use_container_width=True, type="primary"):
    # Build input dict with raw values
    input_dict = {
        "city": city,
        "city_development_index": city_development_index,
        "gender": gender,
        "relevent_experience": relevent_experience,
        "enrolled_university": enrolled_university,
        "education_level": education_level,
        "major_discipline": major_discipline,
        "experience": experience,
        "company_size": company_size,
        "company_type": company_type,
        "last_new_job": last_new_job,
        "training_hours": training_hours,
    }

    input_df = pd.DataFrame([input_dict])

    # Apply the same LabelEncoder used during training for each categorical column
    categorical_cols = [
        'city', 'gender', 'relevent_experience', 'enrolled_university',
        'education_level', 'major_discipline', 'experience',
        'company_size', 'company_type', 'last_new_job'
    ]

    for col in categorical_cols:
        le = le_dict[col]
        val = input_df[col].astype(str).iloc[0]
        # Handle unseen labels gracefully
        if val in le.classes_:
            input_df[col] = le.transform([val])
        else:
            # Fallback: use the most common class (index 0 after fit on training data)
            input_df[col] = le.transform([le.classes_[0]])

    # Ensure column order matches training
    input_df = input_df[feature_cols]

    # Make prediction
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.header("Prediction Result")

    if prediction == 0:
        st.success(f"✅ This employee is **likely to stay**.")
        st.metric("Probability of Staying", f"{proba[0]*100:.1f}%")
    else:
        st.error(f"⚠️ This employee is **likely to look for a new job**.")
        st.metric("Probability of Leaving", f"{proba[1]*100:.1f}%")

    # Confidence bar
    st.markdown("**Prediction Confidence**")
    col_a, col_b = st.columns(2)
    col_a.metric("Stay", f"{proba[0]*100:.1f}%")
    col_b.metric("Leave", f"{proba[1]*100:.1f}%")
    st.progress(float(proba[1]))
