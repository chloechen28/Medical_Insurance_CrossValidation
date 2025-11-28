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

#my prediction is a continous number so accuracy would't work well with it