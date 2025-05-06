import sys, json, joblib, pandas as pd

BUNDLE_PATH = "attrition_model.pkl"
bundle = joblib.load(BUNDLE_PATH)

model      = bundle["model"]
encoder    = bundle["encoder"]
threshold  = bundle.get("thr", 0.62)      # fallback jika key hilang
train_cols = bundle["train_cols"]

# ------------------------------------------------------------
def encode_input(json_str: str) -> pd.DataFrame:
    raw = pd.DataFrame([json.loads(json_str)])

    # --- Revisi: Sinkronisasi kolom input untuk encoder ---
    if hasattr(encoder, "feature_names_in_"):
        expected_input_cols = list(encoder.feature_names_in_)
    else:
        expected_input_cols = list(raw.columns)
    for col in expected_input_cols:
        if col not in raw.columns:
            raw[col] = pd.NA
    encoder_input_df = raw[expected_input_cols]
    # --- Akhir revisi ---

    X_te = encoder.transform(encoder_input_df)

    # Identify original object columns NOT handled by TargetEncoder for OHE
    original_object_cols = raw.select_dtypes("object").columns
    ohe_cols = [c for c in original_object_cols if c not in encoder.cols]

    # Create the final DataFrame starting with target-encoded columns
    X = X_te.copy()

    # Add back the columns intended for OHE from the original raw data
    for col in ohe_cols:
        X[col] = raw[col]

    # Apply One-Hot Encoding to the appropriate columns
    X = pd.get_dummies(X, columns=ohe_cols, drop_first=True)

    # Ensure all final training columns are present, adding missing ones (often from OHE) with 0
    for c in train_cols:
        if c not in X.columns:
            X[c] = 0

    # Return the final DataFrame with columns ordered as per train_cols
    return X[train_cols]

# ------------------------------------------------------------
if __name__ == "__main__":
    dummy_data = [
        {"Age": 30, "BusinessTravel": "Travel_Rarely",   "DailyRate": 1102, "Department": "Sales",  "DistanceFromHome": 1, "Education": 2, "EducationField": "Life Sciences", "EnvironmentSatisfaction": 2, "Gender": "Female", "HourlyRate": 94, "JobInvolvement": 3, "JobLevel": 2, "JobRole": "Sales Executive",     "JobSatisfaction": 4, "MaritalStatus": "Single", "MonthlyIncome": 4000, "MonthlyRate": 15000, "NumCompaniesWorked": 3, "OverTime": "Yes", "PercentSalaryHike": 11, "PerformanceRating": 3, "RelationshipSatisfaction": 1, "StockOptionLevel": 0, "TotalWorkingYears": 5, "TrainingTimesLastYear": 2, "WorkLifeBalance": 1, "YearsAtCompany": 4, "YearsInCurrentRole": 2, "YearsSinceLastPromotion": 0, "YearsWithCurrManager": 3},
        {"Age": 31, "BusinessTravel": "Travel_Frequently","DailyRate": 1200, "Department": "R&D",    "DistanceFromHome": 5, "Education": 3, "EducationField": "Medical",        "EnvironmentSatisfaction": 3, "Gender": "Male",   "HourlyRate": 80, "JobInvolvement": 2, "JobLevel": 1, "JobRole": "Lab Technician",      "JobSatisfaction": 2, "MaritalStatus": "Married","MonthlyIncome": 5000, "MonthlyRate": 17000, "NumCompaniesWorked": 1, "OverTime": "No",  "PercentSalaryHike": 14, "PerformanceRating": 4, "RelationshipSatisfaction": 2, "StockOptionLevel": 1, "TotalWorkingYears": 6, "TrainingTimesLastYear": 1, "WorkLifeBalance": 2, "YearsAtCompany": 5, "YearsInCurrentRole": 3, "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 2},
        {"Age": 32, "BusinessTravel": "Travel_Rarely",   "DailyRate": 1300, "Department": "HR",    "DistanceFromHome": 2, "Education": 4, "EducationField": "Human Resources","EnvironmentSatisfaction": 4, "Gender": "Female", "HourlyRate": 85, "JobInvolvement": 4, "JobLevel": 3, "JobRole": "HR Manager",         "JobSatisfaction": 3, "MaritalStatus": "Single","MonthlyIncome": 6000, "MonthlyRate": 18000, "NumCompaniesWorked": 4, "OverTime": "Yes", "PercentSalaryHike": 13, "PerformanceRating": 3, "RelationshipSatisfaction": 3, "StockOptionLevel": 2, "TotalWorkingYears": 7, "TrainingTimesLastYear": 2, "WorkLifeBalance": 3, "YearsAtCompany": 6, "YearsInCurrentRole": 4, "YearsSinceLastPromotion": 2, "YearsWithCurrManager": 4},
        {"Age": 33, "BusinessTravel": "Non-Travel",     "DailyRate": 1400, "Department": "Sales", "DistanceFromHome": 10,"Education": 1, "EducationField": "Marketing",       "EnvironmentSatisfaction": 1, "Gender": "Male",   "HourlyRate": 75, "JobInvolvement": 1, "JobLevel": 1, "JobRole": "Sales Representative","JobSatisfaction": 1, "MaritalStatus": "Divorced","MonthlyIncome": 4500, "MonthlyRate": 16000, "NumCompaniesWorked": 2, "OverTime": "No",  "PercentSalaryHike": 12, "PerformanceRating": 2, "RelationshipSatisfaction": 1, "StockOptionLevel": 0, "TotalWorkingYears": 8, "TrainingTimesLastYear": 1, "WorkLifeBalance": 2, "YearsAtCompany": 7, "YearsInCurrentRole": 5, "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 5},
        {"Age": 34, "BusinessTravel": "Travel_Frequently","DailyRate": 1500, "Department": "R&D",   "DistanceFromHome": 3, "Education": 3, "EducationField": "Life Sciences", "EnvironmentSatisfaction": 2, "Gender": "Female", "HourlyRate": 90, "JobInvolvement": 2, "JobLevel": 2, "JobRole": "Research Scientist",  "JobSatisfaction": 4, "MaritalStatus": "Married","MonthlyIncome": 5500, "MonthlyRate": 17500, "NumCompaniesWorked": 5, "OverTime": "Yes", "PercentSalaryHike": 15, "PerformanceRating": 4, "RelationshipSatisfaction": 2, "StockOptionLevel": 1, "TotalWorkingYears": 9, "TrainingTimesLastYear": 3, "WorkLifeBalance": 3, "YearsAtCompany": 8, "YearsInCurrentRole": 6, "YearsSinceLastPromotion": 2, "YearsWithCurrManager": 6},
        {"Age": 35, "BusinessTravel": "Travel_Rarely",   "DailyRate": 1600, "Department": "HR",   "DistanceFromHome": 4, "Education": 2, "EducationField": "Other",           "EnvironmentSatisfaction": 3, "Gender": "Male",   "HourlyRate":100, "JobInvolvement": 3, "JobLevel": 3, "JobRole": "HR Specialist",       "JobSatisfaction": 2, "MaritalStatus": "Single","MonthlyIncome": 6500, "MonthlyRate": 18500, "NumCompaniesWorked": 6, "OverTime": "No",  "PercentSalaryHike": 16, "PerformanceRating": 3, "RelationshipSatisfaction": 3, "StockOptionLevel": 2, "TotalWorkingYears":10, "TrainingTimesLastYear": 2, "WorkLifeBalance": 4, "YearsAtCompany": 9, "YearsInCurrentRole": 7, "YearsSinceLastPromotion": 3, "YearsWithCurrManager": 7},
        {"Age": 36, "BusinessTravel": "Non-Travel",     "DailyRate": 1700, "Department": "Sales", "DistanceFromHome": 6, "Education": 1, "EducationField": "Marketing",       "EnvironmentSatisfaction": 4, "Gender": "Female", "HourlyRate": 95, "JobInvolvement": 4, "JobLevel": 4, "JobRole": "Sales Manager",      "JobSatisfaction": 3, "MaritalStatus": "Married","MonthlyIncome": 7000, "MonthlyRate": 19000, "NumCompaniesWorked": 7, "OverTime": "Yes", "PercentSalaryHike": 17, "PerformanceRating": 4, "RelationshipSatisfaction": 4, "StockOptionLevel": 3, "TotalWorkingYears":11, "TrainingTimesLastYear": 3, "WorkLifeBalance": 4, "YearsAtCompany":10, "YearsInCurrentRole": 8, "YearsSinceLastPromotion": 4, "YearsWithCurrManager": 8},
        {"Age": 37, "BusinessTravel": "Travel_Frequently","DailyRate": 1800, "Department": "R&D",   "DistanceFromHome": 7, "Education": 4, "EducationField": "Medical",        "EnvironmentSatisfaction": 1, "Gender": "Male",   "HourlyRate":105, "JobInvolvement": 1, "JobLevel": 1, "JobRole": "Research Director",   "JobSatisfaction": 2, "MaritalStatus": "Divorced","MonthlyIncome": 7500, "MonthlyRate": 19500, "NumCompaniesWorked": 8, "OverTime": "No",  "PercentSalaryHike": 18, "PerformanceRating": 2, "RelationshipSatisfaction": 1, "StockOptionLevel": 3, "TotalWorkingYears":12, "TrainingTimesLastYear": 4, "WorkLifeBalance": 2, "YearsAtCompany":11, "YearsInCurrentRole": 9, "YearsSinceLastPromotion": 5, "YearsWithCurrManager": 9},
        {"Age": 38, "BusinessTravel": "Travel_Rarely",   "DailyRate": 1900, "Department": "HR",   "DistanceFromHome": 8, "Education": 2, "EducationField": "Human Resources","EnvironmentSatisfaction": 2, "Gender": "Female", "HourlyRate":110, "JobInvolvement": 2, "JobLevel": 2, "JobRole": "HR Coordinator",      "JobSatisfaction": 4, "MaritalStatus": "Single","MonthlyIncome": 8000, "MonthlyRate": 20000, "NumCompaniesWorked": 9, "OverTime": "Yes", "PercentSalaryHike": 19, "PerformanceRating": 3, "RelationshipSatisfaction": 2, "StockOptionLevel": 1, "TotalWorkingYears":13, "TrainingTimesLastYear": 4, "WorkLifeBalance": 3, "YearsAtCompany":12, "YearsInCurrentRole":10, "YearsSinceLastPromotion": 6, "YearsWithCurrManager":10},
        {"Age": 39, "BusinessTravel": "Non-Travel",     "DailyRate": 2000, "Department": "Sales", "DistanceFromHome": 9, "Education": 3, "EducationField": "Marketing",       "EnvironmentSatisfaction": 3, "Gender": "Male",   "HourlyRate":115, "JobInvolvement": 3, "JobLevel": 3, "JobRole": "Sales Executive",    "JobSatisfaction": 1, "MaritalStatus": "Married","MonthlyIncome": 8500, "MonthlyRate": 20500, "NumCompaniesWorked":10, "OverTime": "No",  "PercentSalaryHike": 20, "PerformanceRating": 4, "RelationshipSatisfaction": 3, "StockOptionLevel": 2, "TotalWorkingYears":14, "TrainingTimesLastYear": 5, "WorkLifeBalance": 4, "YearsAtCompany":13, "YearsInCurrentRole":11, "YearsSinceLastPromotion": 7, "YearsWithCurrManager":11}
    ]
    for idx, record in enumerate(dummy_data, 1):
        X_enc = encode_input(json.dumps(record))
        prob = model.predict_proba(X_enc)[:, 1][0]
        pred = "LEAVE" if prob >= threshold else "STAY"
        print(f"Record {idx}: P(Attrition) = {prob:.3f}, Predicted = {pred}")