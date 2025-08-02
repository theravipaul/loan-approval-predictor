import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import subprocess

MODEL_PATH = "models/final_loan_model.pkl"

# Train automatically if model not found
if not os.path.exists(MODEL_PATH):
    st.info("Model not found. Training it now...")
    subprocess.run(["python", "train_model.py"])

# Load trained model
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Prediction App")

col1, col2 = st.columns(2)
with col1:
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    Property_Area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])
    Dependents_num = st.number_input("Number of Dependents", min_value=0, max_value=5, step=1)

with col2:
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
    LoanAmount = st.number_input("Loan Amount", min_value=0)
    Loan_Amount_Term = st.number_input("Loan Term (Months)", min_value=12, step=12)
    Credit_History = st.selectbox("Credit History", [0, 1])

TotalIncome = ApplicantIncome + CoapplicantIncome
DebtIncomeRatio = LoanAmount / TotalIncome if TotalIncome > 0 else 0
LoanAmountLog = np.log1p(LoanAmount)
LoanTermYears = Loan_Amount_Term / 12

input_data = pd.DataFrame([{
    'Gender': Gender,
    'Married': Married,
    'Education': Education,
    'Self_Employed': Self_Employed,
    'Property_Area': Property_Area,
    'LoanAmountLog': LoanAmountLog,
    'TotalIncome': TotalIncome,
    'DebtIncomeRatio': DebtIncomeRatio,
    'LoanTermYears': LoanTermYears,
    'Credit_History': Credit_History,
    'Dependents_num': Dependents_num
}])

if st.button("Predict Loan Status"):
    prediction = model.predict(input_data)[0]
    result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"
    st.subheader(result)
