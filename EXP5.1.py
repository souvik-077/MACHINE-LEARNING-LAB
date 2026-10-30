
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = [
    [1, 35], [2, 40], [2, 50], [3, 45], [3, 55],
    [4, 40], [4, 50], [4, 60], [5, 45], [5, 55],
    [5, 60], [6, 45], [6, 50], [7, 45], [7, 50],
    [6, 60], [6, 65], [7, 60], [7, 70], [8, 50],
    [8, 60], [8, 70], [9, 55], [9, 65], [9, 75],
    [10, 55], [10, 65], [10, 75], [11, 60], [11, 80]
]

y = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1
]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("--- Student Pass/Fail Prediction ---")
print("Accuracy:", accuracy_score(y_test, y_pred))

new_student = [[8, 75]]
new_student = scaler.transform(new_student)

result = model.predict(new_student)

if result[0] == 1:
    print("Student will Pass")
else:
    print("Student will Fail")
