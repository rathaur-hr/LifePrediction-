import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and encoders
model = joblib.load("life_expectancy_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("🌍 Life Expectancy Predictor")

# Input fields
country = st.text_input("Country")
year = st.number_input("Year", min_value=1900, max_value=2100, value=2015)
status = st.selectbox("Status", ["Developing", "Developed"])
adult_mortality = st.number_input("Adult Mortality")
infant_deaths = st.number_input("Infant Deaths")
alcohol = st.number_input("Alcohol Consumption")
percentage_expenditure = st.number_input("Percentage Expenditure")
hepatitis_b = st.number_input("Hepatitis B")
measles = st.number_input("Measles")
bmi = st.number_input("BMI")
under_five_deaths = st.number_input("Under-Five Deaths")
polio = st.number_input("Polio")
total_expenditure = st.number_input("Total Expenditure")
diphtheria = st.number_input("Diphtheria")
hiv_aids = st.number_input("HIV/AIDS")
gdp = st.number_input("GDP")
population = st.number_input("Population")
thinness_1_19 = st.number_input("Thinness 1-19 years")
thinness_5_9 = st.number_input("Thinness 5-9 years")
income_composition = st.number_input("Income Composition of Resources")
schooling = st.number_input("Schooling")

# Prepare input
input_data = {
    'Country': label_encoders['Country'].transform([country])[0] if country in label_encoders['Country'].classes_ else 0,
    'Year': year,
    'Status': label_encoders['Status'].transform([status])[0],
    'Adult Mortality': adult_mortality,
    'infant deaths': infant_deaths,
    'Alcohol': alcohol,
    'percentage expenditure': percentage_expenditure,
    'Hepatitis B': hepatitis_b,
    'Measles ': measles,
    ' BMI ': bmi,
    'under-five deaths ': under_five_deaths,
    'Polio': polio,
    'Total expenditure': total_expenditure,
    'Diphtheria ': diphtheria,
    ' HIV/AIDS': hiv_aids,
    'GDP': gdp,
    'Population': population,
    ' thinness  1-19 years': thinness_1_19,
    ' thinness 5-9 years': thinness_5_9,
    'Income composition of resources': income_composition,
    'Schooling': schooling
}

input_df = pd.DataFrame([input_data])

# Predict
if st.button("Predict Life Expectancy"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Life Expectancy: {prediction:.2f} years")

