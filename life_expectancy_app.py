import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("life_expectancy_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("🌍 Life Expectancy Predictor")

with st.form("life_expectancy_form"):
    country = st.text_input("Country", help="Enter the name of the country")
    year = st.number_input("Year", min_value=1900, max_value=2100, value=2015, help="Enter the year of prediction")
    status = st.selectbox("Country Status", ["Developing", "Developed"], help="Select the development status of the country")
    adult_mortality = st.number_input("Adult Mortality", help="Mortality rate of adults per 1000 population")
    infant_deaths = st.number_input("Infant Deaths", help="Number of infant deaths per 1000 births")
    alcohol = st.number_input("Alcohol Consumption", help="Alcohol consumption per capita (liters)")
    percentage_expenditure = st.number_input("Percentage Expenditure", help="Expenditure on health as percentage of GDP")
    hepatitis_b = st.number_input("Hepatitis B Immunization (%)", help="Percentage of people immunized for Hepatitis B")
    measles = st.number_input("Measles Cases", help="Number of reported measles cases")
    bmi = st.number_input("Body Mass Index (BMI)", help="Average BMI of the population")
    under_five_deaths = st.number_input("Under-Five Deaths", help="Number of deaths under age 5 per 1000 births")
    polio = st.number_input("Polio Immunization (%)", help="Percentage of people immunized for Polio")
    total_expenditure = st.number_input("Total Health Expenditure (%)", help="Total health expenditure as percentage of GDP")
    diphtheria = st.number_input("Diphtheria Immunization (%)", help="Percentage of people immunized for Diphtheria")
    hiv_aids = st.number_input("HIV/AIDS Cases", help="Number of HIV/AIDS cases per 1000 population")
    gdp = st.number_input("GDP", help="Gross Domestic Product per capita")
    population = st.number_input("Population", help="Total population of the country")
    thinness_1_19 = st.number_input("Thinness (1-19 years)", help="Prevalence of thinness among 1-19 year olds")
    thinness_5_9 = st.number_input("Thinness (5-9 years)", help="Prevalence of thinness among 5-9 year olds")
    income_composition = st.number_input("Income Composition of Resources", help="HDI income composition index (0–1)")
    schooling = st.number_input("Schooling (Years)", help="Average number of years of schooling")

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
                ' thinness 1-19 years': thinness_1_19,
                ' thinness 5-9 years': thinness_5_9,
                'Income composition of resources': income_composition,
                'Schooling': schooling
            }

            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]

            st.success(f"Predicted Life Expectancy: {prediction:.2f} years")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
