
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

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

y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, threshold = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

print("--- ROC Curve and AUC ---")
print("AUC Score:", auc)

plt.plot(fpr, tpr, label="Logistic Regression")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
