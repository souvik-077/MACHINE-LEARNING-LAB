import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

data = {
    "Age": [20, 21, None, 23, 24],
    "Salary": [25000, 30000, 28000, None, 40000],
    "Department": ["IT", "HR", "IT", "Sales", "HR"],
    "Years_of_Experience": [1, 2, 3, None, 5]
}

df = pd.DataFrame(data)

numeric_features = ["Age", "Salary", "Years_of_Experience"]

data_numeric = df[numeric_features]

standard_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

minmax_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

standard_data = standard_pipeline.fit_transform(data_numeric)
minmax_data = minmax_pipeline.fit_transform(data_numeric)

print("--- StandardScaler Result ---")
print(standard_data)

print("\n--- MinMaxScaler Result ---")
print(minmax_data)

print("\n--- Numerical Range ---")

print("StandardScaler Minimum:", standard_data.min())
print("StandardScaler Maximum:", standard_data.max())

print("MinMaxScaler Minimum:", minmax_data.min())
print("MinMaxScaler Maximum:", minmax_data.max())
