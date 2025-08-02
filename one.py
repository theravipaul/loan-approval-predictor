import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

MODEL_FILE = "models/final_loan_model.pkl"
DATA_FILE = "data/loan-train.csv"

# ----------------- Build Pipeline -----------------
def build_pipeline(numerical_columns, categorical_columns):
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ])

    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=min(6, len(numerical_columns) + len(categorical_columns)))),
        ('model', RandomForestClassifier(random_state=42))
    ])

    return full_pipeline

# ----------------- Train Model -----------------
def train_model():
    st.info("Training model... please wait ⏳")

    df = pd.read_csv(DATA_FILE)

    # Feature engineering
    df['Credit_History'] = df['Credit_History'].fillna(0).astype(int)
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['DebtIncomeRatio'] = df['LoanAmount'] / df['TotalIncome']
    df['LoanAmountLog'] = np.log1p(df['LoanAmount'])
    df['LoanTermYears'] = df['Loan_Amount_Term'] / 12

    df['Dependents'] = df['Dependents'].astype(str).replace('nan', '0')
    df['Dependents_num'] = df['Dependents'].replace('3+', '3').astype(int)

    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    df_label = df['Loan_Status']
    df_features = df.drop(['Loan_ID', 'Dependents', 'ApplicantIncome', 'CoapplicantIncome',
                           'Loan_Amount_Term', 'Loan_Status'], axis=1)

    numerical_columns = [
        'LoanAmountLog', 'TotalIncome', 'DebtIncomeRatio',
        'LoanTermYears', 'Credit_History', 'Dependents_num'
    ]
    categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

    x_train, x_test, y_train, y_test = train_test_split(df_features, df_label, test_size=0.2, random_state=42)

    pipeline = build_pipeline(numerical_columns, categorical_columns)

    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 5, 10]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x_train, y_train)

    best_pipeline = build_pipeline(numerical_columns, categorical_columns)
    best_pipeline.set_params(**grid_search.best_params_)
    best_pipeline.fit(df_features, df_label)

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipeline, MODEL_FILE)

    st.success(f"Model trained successfully ✅ (Accuracy: {best_pipeline.score(x_test, y_test):.2f})")
    return best_pipeline

# ----------------- Load or Train -----------------
if os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)
    except:
        model = train_model()
else:
    model = train_model()

# ----------------- Streamlit UI -----------------
st.title("🏦 Loan Approval Predictor")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=1000000, step=1000)
loan_term = st.number_input("Loan Term (in days)", min_value=30, max_value=500, step=10)
credit_history = st.selectbox("Credit History", [0, 1])
dependents = st.selectbox("Dependents", [0, 1, 2, 3])
applicant_income = st.number_input("Applicant Income", min_value=1000, max_value=100000, step=100)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0, max_value=100000, step=100)

# Feature Engineering
total_income = applicant_income + coapplicant_income
debt_income_ratio = loan_amount / total_income if total_income > 0 else 0
loan_amount_log = np.log1p(loan_amount)
loan_term_years = loan_term / 12

input_data = pd.DataFrame([{
    "Gender": gender,
    "Married": married,
    "Education": education,
    "Self_Employed": self_employed,
    "Property_Area": property_area,
    "LoanAmount": loan_amount,
    "TotalIncome": total_income,
    "DebtIncomeRatio": debt_income_ratio,
    "LoanAmountLog": loan_amount_log,
    "LoanTermYears": loan_term_years,
    "Credit_History": credit_history,
    "Dependents_num": dependents
}])

if st.button("Predict Loan Approval"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
