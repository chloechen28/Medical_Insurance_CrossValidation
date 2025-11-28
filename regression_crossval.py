import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("medical_insurance.csv")
X = df[['age', 'household_size', 'visits_last_year', 
        'hospitalizations_last_3yrs', 'days_hospitalized_last_3yrs',
        'medication_count', 'systolic_bp', 'diastolic_bp', 
        'ldl', 'hba1c', 'chronic_count', 
        'hypertension', 'diabetes', 'asthma', 'copd',
        'cardiovascular_disease', 'cancer_history', 'kidney_disease',
        'liver_disease', 'arthritis', 'mental_health', 'risk_score', 
        'bmi', 'deductible', 'income', 'policy_term_years', 'dependents']] #factors that influence annual medical cost

y = df['annual_medical_cost'] #wants to predict annual medical cost

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

print(model.intercept_)
print(model.coef_)

y_pred = model.predict(X_test)

predictions = pd.DataFrame({'Actual': y_test, 'Predicted' : y_pred})
print(predictions.head())

r2 = r2_score(y_test, y_pred)
print(f"R-squared score: {r2}")

#LOOCV
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
loo_errors = []
counter = 0
for train_idx, val_idx in loo.split(X):
        counter = counter + 1
        if counter > 4000: #takes too long csv file has over 100,000 + rows
                 break
        print("Looping: ", counter)
        X_train_loo, X_val_loo = X.iloc[train_idx], X.iloc[val_idx]
        y_train_loo, y_val_loo = y.iloc[train_idx], y.iloc[val_idx]

        model_loo = LinearRegression()
        model_loo.fit(X_train_loo, y_train_loo)

        pred = model_loo.predict(X_val_loo)
        loo_errors.append((y_val_loo.values[0] - pred[0]) ** 2)
# Calculate LOOCV MSE and RMSE
loo_mse = np.mean(loo_errors)
loo_rmse = np.sqrt(loo_mse)

print(f"LOOCV MSE: {loo_mse:.4f}")
print(f"LOOCV RMSE: {loo_rmse:.4f}")

# kfold validation
from sklearn.model_selection import KFold 
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation
kf_errors = []

for train_idx, val_idx in kf.split(X):
    X_train_kf, X_val_kf = X.iloc[train_idx], X.iloc[val_idx]
    y_train_kf, y_val_kf = y.iloc[train_idx], y.iloc[val_idx]

    model_kf = LinearRegression()
    model_kf.fit(X_train_kf, y_train_kf)

    preds = model_kf.predict(X_val_kf)
    kf_error = mean_squared_error(y_val_kf, preds)  # MSE for K-Fold
    kf_errors.append(kf_error)

kf_mse = np.mean(kf_errors)
kf_rmse = np.sqrt(kf_mse)

print(f"K-Fold CV MSE: {kf_mse:.4f}")
print(f"K-Fold CV RMSE: {kf_rmse:.4f}")
