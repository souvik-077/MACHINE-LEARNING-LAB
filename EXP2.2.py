import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)

plt.figure(figsize=(12, 6))
sns.boxplot(data=df)

plt.title("Boxplots of Wine Dataset")
plt.xticks(rotation=90)

plt.show()
