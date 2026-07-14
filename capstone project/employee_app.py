import streamlit as st
import pandas as pd
from joblib import load

# Load the trained Gradient Boosting model and feature columns
model = load('gb_model.joblib')
feature_cols = load('feature_cols.joblib')

# Create a Streamlit app
st.title("Employee Retention Prediction App")

# Input fields
st.header("Enter Employee Information")
city = st.text_input("City Code", value="city_103")
city_development_index = st.slider("City Development Index", 0.0, 1.0, 0.92)
gender = st.selectbox("Gender", ("Male", "Female", "Other"))
relevent_experience = st.selectbox("Relevant Experience", ("Has relevent experience", "No relevent experience"))
enrolled_university = st.selectbox("Enrolled University", ("no_enrollment", "Full time course", "Part time course"))
education_level = st.selectbox("Education Level", ("Graduate", "Masters", "Phd", "High School", "Primary School"))
major_discipline = st.selectbox("Major Discipline", ("STEM", "Business Degree", "Arts", "Humanities", "No Major", "Other"))
experience = st.selectbox("Experience (years)", ["<1"] + [str(i) for i in range(1, 21)] + [">20"])
company_size = st.selectbox("Company Size", ("<10", "10/49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"))
company_type = st.selectbox("Company Type", ("Pvt Ltd", "Funded Startup", "Public Sector", "Early Stage Startup", "NGO", "Other"))
last_new_job = st.selectbox("Years Since Last Job Change", ("never", "1", "2", "3", "4", ">4"))
training_hours = st.number_input("Training Hours", min_value=0, max_value=400, value=50)

# Build input dataframe
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
    "training_hours": training_hours
}

input_df = pd.DataFrame([input_dict])
input_encoded = pd.get_dummies(input_df, drop_first=True)
input_encoded = input_encoded.reindex(columns=feature_cols, fill_value=0)

# Make prediction
prediction = model.predict(input_encoded)

# Display result
st.header("Prediction Result")
if prediction[0] == 0:
    st.success("This employee is likely to stay.")
else:
    st.error("This employee is likely to look for a new job.")
