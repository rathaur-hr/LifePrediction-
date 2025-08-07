import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Define model and encoder paths
model_path = "life_expectancy_model.pkl"
encoders_path = "label_encoders.pkl"

# Check if model and encoders exist
if not os.path.exists(model_path) or not os.path.exists(encoders_path):
    st.error("Model or encoder files not found. Ensure 'life_expectancy_model.pkl' and 'label_encoders.pkl' are in the same directory.")
else:
    # Load model and encoders
    model = joblib.load(model_path)
    label_encoders = joblib.load(encoders_path)

    st.title("🌍 Life Expectancy Predictor")

    with st.form("life_expectancy_form"):
        st.subheader("Enter Required Information")

        country = st.text_input("Country", help="Enter the name of the country")
        year = st.number_input("Year", min_value=1900, max_value=2100, value=2015, help="Year of prediction")
        status = st.selectbox("Development Status", ["Developing", "Developed"], help="Select the country's development status")
        adult_mortality = st.number_input("Adult Mortality", min_value=0.0, help="Mortality rate of adults")
        infant_deaths = st.number_input("Infant Deaths", min_value=0.0, help="Number of infant deaths per 1000 births")
        alcohol = st.number_input("Alcohol Consumption", min_value=0.0, help="Alcohol consumption per capita (liters)")
        hiv_aids = st.number_input("HIV/AIDS", min_value=0.0, help="Deaths per 1000 live births due to HIV/AIDS")
        bmi = st.number_input("BMI", min_value=0.0, help="Average Body Mass Index")
        gdp = st.number_input("GDP", min_value=0.0, help="Gross Domestic Product")
        income_composition = st.number_input("Income Composition of Resources", min_value=0.0, max_value=1.0, help="HDI component")
        population = st.number_input("Population", min_value=0.0, help="Population of the country")
        schooling = st.number_input("Schooling", min_value=0.0, help="Average years of schooling")

        submitted = st.form_submit_button("Predict Life Expectancy")

    if submitted:
        if not country:
            st.error("Please enter the country name.")
        else:
            try:
                input_data = {
                    'Country': label_encoders['Country'].transform([country])[0]
                    if country in label_encoders['Country'].classes_ else 0,
                    'Year': year,
                    'Status': label_encoders['Status'].transform([status])[0],
                    'Adult Mortality': adult_mortality,
                    'infant deaths': infant_deaths,
                    'Alcohol': alcohol,
                    ' HIV/AIDS': hiv_aids,
                    ' BMI ': bmi,
                    'GDP': gdp,
                    'Income composition of resources': income_composition,
                    'Population': population,
                    'Schooling': schooling
                }

                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                st.success(f"Predicted Life Expectancy: {prediction:.2f} years")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
