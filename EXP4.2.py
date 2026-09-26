import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("real_estate_data.csv")

print("--- First Five Rows ---")
print(df.head())

X = df[["size", "bedrooms"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Evaluation Metrics ---")
print("MAE:", mae)
print("MSE:", mse)
print("R2:", r2)

print("\n--- Coefficients ---")
print("Size Coefficient:", model.coef_[0])
print("Bedrooms Coefficient:", model.coef_[1])
print("Intercept:", model.intercept_)

print("\n--- Actual vs Predicted Prices ---")
print("Actual:", y_test.values)
print("Predicted:", y_pred)