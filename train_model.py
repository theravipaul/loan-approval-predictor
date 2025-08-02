import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

MODEL_PATH = "models/final_loan_model.pkl"

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
        ('pca', PCA(n_components=6)),
        ('model', RandomForestClassifier(random_state=42))
    ])
    return full_pipeline

def main():
    print("🚀 Training Loan Approval Model...")
    os.makedirs("models", exist_ok=True)

    data = pd.read_csv("loan-train.csv")
    df = data.copy()

    # Feature Engineering
    df['Credit_History'] = df['Credit_History'].fillna(0).astype(int)
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['DebtIncomeRatio'] = df['LoanAmount'] / df['TotalIncome']
    df['LoanAmountLog'] = np.log1p(df['LoanAmount'])
    df['LoanTermYears'] = df['Loan_Amount_Term'] / 12
    df['Dependents'] = df['Dependents'].astype(str).replace('nan', '0')
    df['Dependents_num'] = df['Dependents'].replace('3+', '3').astype(int)
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    df_label = df['Loan_Status']
    df_features = df.drop(['Loan_ID', 'Dependents', 'ApplicantIncome',
                           'CoapplicantIncome', 'Loan_Amount_Term', 'Loan_Status'], axis=1)

    numerical_columns = ['LoanAmountLog', 'TotalIncome', 'DebtIncomeRatio',
                         'LoanTermYears', 'Credit_History', 'Dependents_num']
    categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

    x_train, x_test, y_train, y_test = train_test_split(df_features, df_label, test_size=0.2, random_state=42)
    
    pipeline = build_pipeline(numerical_columns, categorical_columns)
    
    # GridSearchCV
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 5, 10],
        'model__min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x_train, y_train)

    print("✅ Best Parameters:", grid_search.best_params_)
    print("✅ Cross-Validation Accuracy:", grid_search.best_score_)

    # Final model
    best_pipeline = build_pipeline(numerical_columns, categorical_columns)
    best_pipeline.set_params(**grid_search.best_params_)
    best_pipeline.fit(df_features, df_label)

    print("🎯 Final Model Accuracy:", best_pipeline.score(x_test, y_test))
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
