
# Exploratory Data Analysis

import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda(data):
    print("\nprint of data: ", data.shape)
    print("\nFirst 5 rows: \n", data.head())
    print("\nMissing values: \n", data.isnull().sum())

    # Distribution of target variable
    sns.countplot(x="quality", data=data)
    plt.title("Wine Quality Distribution")
    plt.show()

    # Heatmap for correlations
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()