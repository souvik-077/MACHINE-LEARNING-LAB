import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

data = {
    "Age": [20, 21, None, 23, 24],
    "Salary": [25000, 30000, 28000, None, 40000],
    "Department": ["IT", "HR", "IT", "Sales", "HR"],
    "Years_of_Experience": [1, 2, 3, None, 5]
}

df = pd.DataFrame(data)

print("--- Original Dataset ---")
print(df)

numeric_features = ["Age", "Salary", "Years_of_Experience"]
categorical_features = ["Department"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

X = preprocessor.fit_transform(df)

print("\n--- MinMaxScaler Result ---")
print(X)
