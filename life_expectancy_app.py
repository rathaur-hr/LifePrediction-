import streamlit as st
import pandas as pd
import joblib

# Load the model and encoders
model = joblib.load("life_expectancy_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("🌍 Life Expectancy Predictor")

with st.form("prediction_form"):
    st.subheader("Please enter all required details:")

    country = st.text_input("Country Name", help="Type the exact country name.")
    year = st.number_input("Prediction Year", min_value=1900, max_value=2100, value=2015, help="Year for which you want to predict.")
    status = st.selectbox("Is the Country Developed or Developing?", ["Developing", "Developed"])

    adult_mortality = st.number_input("Adult Mortality Rate (per 1000)", min_value=0.0)
    infant_deaths = st.number_input("Infant Deaths (per 1000 births)", min_value=0.0)
    alcohol = st.number_input("Alcohol Consumption (liters per capita)", min_value=0.0)
    percentage_expenditure = st.number_input("Health Expenditure (% of GDP)", min_value=0.0)
    hepatitis_b = st.number_input("Hepatitis B Immunization Coverage (%)", min_value=0.0, max_value=100.0)
    measles = st.number_input("Reported Measles Cases", min_value=0.0)
    bmi = st.number_input("Average BMI", min_value=0.0)
    under_five_deaths = st.number_input("Under-5 Deaths (per 1000 births)", min_value=0.0)
    polio = st.number_input("Polio Immunization Coverage (%)", min_value=0.0, max_value=100.0)
    total_expenditure = st.number_input("Total Government Health Spending (% of total)", min_value=0.0)
    diphtheria = st.number_input("Diphtheria Immunization Coverage (%)", min_value=0.0, max_value=100.0)
    hiv_aids = st.number_input("HIV/AIDS Deaths (per 1000)", min_value=0.0)
    gdp = st.number_input("GDP per Capita (USD)", min_value=0.0)
    population = st.number_input("Population", min_value=0.0)
    thinness_1_19 = st.number_input("Thinness Prevalence (Ages 10–19)", min_value=0.0)
    thinness_5_9 = st.number_input("Thinness Prevalence (Ages 5–9)", min_value=0.0)
    income_composition = st.number_input("Income Index (HDI Component)", min_value=0.0, max_value=1.0)
    schooling = st.number_input("Schooling (avg yrs)", min_value=0.0)

    submitted = st.form_submit_button("Predict Life Expectancy")

if submitted:
    # Validate all fields are filled
    if not country.strip():
        st.error("Country name is required.")
    else:
        try:
            # Transform categorical inputs
            encoded_country = label_encoders['Country'].transform([country])[0] if country in label_encoders['Country'].classes_ else 0
            encoded_status = label_encoders['Status'].transform([status])[0]

            # Construct input dictionary (keys must match training features, even if names look odd)
            input_data = {
                'Country': encoded_country,
                'Year': year,
                'Status': encoded_status,
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

            st.success(f"🎯 Predicted Life Expectancy: {prediction:.2f} years")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
