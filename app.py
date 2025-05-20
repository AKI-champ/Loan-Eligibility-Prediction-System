import streamlit as st
import pickle
import pandas as pd

st.title("Loan Eligibility Prediction System")

try:
    model_bundle = pickle.load(open("loan_pipeline.pkl", "rb"))
    model = model_bundle["model"]
    expected_features = model_bundle["feature_names"]
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    st.stop()


age = st.number_input("Enter your Age", min_value=18, max_value=100)
gender = st.selectbox("Select your Gender", ["Male", "Female"])
education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "Associate", "Doctorate"])
income = st.number_input("Monthly Income (in ₹)", min_value=0)
home = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
loan_amount = st.number_input("Loan Amount Required (in ₹)", min_value=0)
loan_interest_rate = st.selectbox("Loan Interest Rate (%)", list(range(6, 21)), index=4)
credit_score = st.number_input("Credit Score", min_value=0, max_value=100000)
default_history = st.selectbox("Any Previous Loan Default?", ["No", "Yes"])


loan_percent_income = loan_amount / income if income != 0 else 0

gender_encoded = 1 if gender == "Male" else 0

education_map = {
    "High School": 0,
    "Bachelor": 1,
    "Master": 2,
    "Associate": 3,
    "Doctorate": 4
}
education_encoded = education_map[education]

home_map = {
    "RENT": 0,
    "MORTGAGE": 1,
    "OWN": 2,
    "OTHER": 3
}
home_encoded = home_map[home]

default_encoded = 1 if default_history == "Yes" else 0

user_input = {
    "person_age": int(age),
    "person_income": int(income),
    "person_gender": gender_encoded,
    "person_education": education_encoded,
    "person_home_ownership": home_encoded,
    "loan_amnt": int(loan_amount),
    "loan_int_rate": int(loan_interest_rate),
    "credit_score": int(credit_score),
    "previous_loan_defaults_on_file": default_encoded,
    "loan_percent_income": loan_percent_income
}

user_df = pd.DataFrame([user_input])
user_df = user_df.reindex(columns=expected_features)  

if st.button("Check Eligibility"):
    try:
        prediction = model.predict(user_df)
        proba = model.predict_proba(user_df)[0][1]

        if prediction[0] == 1:
            st.success("You are Eligible for the Loan")
        else:
            st.error("You are Not Eligible for the Loan")

    except Exception as e:
        st.error(f"Prediction failed: {e}")


loan_term_months = st.number_input("Loan Term (in months)", min_value=1)

if st.button("Calculate EMI"):
    P = loan_amount
    R = loan_interest_rate / 12 / 100 
    N = loan_term_months

    if P <= 0 or R <= 0 or N <= 0:
        st.warning("Please make sure all values are greater than zero.")
    else:
        try:
            emi = (P * R * (1 + R) ** N) / ((1 + R) ** N - 1)
            st.info(f"Your Monthly EMI is: ₹{emi:.2f}")
            st.info(f"Your total Amount will be : {emi*loan_term_months:.2f}")
            st.info(f"Total Interest Payable: ₹{(emi * loan_term_months - loan_amount):.2f}")
        except Exception as e:
            st.error(f"Failed to calculate EMI: {e}")

