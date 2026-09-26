import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)

correlation = df.corr()

print("----Correlation Matrix----")
print(correlation)

plt.figure(figsize=(10, 7))
sns.heatmap(correlation, annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")

plt.show()

max_correlation = 0
feature1 = ""
feature2 = ""

for i in range(len(correlation.columns)):
    for j in range(i + 1, len(correlation.columns)):
        if correlation.iloc[i, j] > max_correlation:
            max_correlation = correlation.iloc[i, j]
            feature1 = correlation.columns[i]
            feature2 = correlation.columns[j]

print("\n----Strongest Positive Correlation----")
print(feature1, "and", feature2)
print("Correlation:", max_correlation)