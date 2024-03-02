import pandas as pd                                              # Data manipulation

import seaborn as sns                                            # Visualization

import matplotlib.pyplot as plt                                  # Basic graphics and additional customizations   

from sklearn.feature_selection import SelectKBest, f_classif 

# Data upload
df = pd.read_csv("drinking_water_potability.csv")

# Descriptive statistics
descriptive_stats = df.describe()
print(descriptive_stats)

# Análisis de valores faltantes
missing_values = df.isnull().sum()
percentage_missing = (missing_values / len(df)) * 100
print(percentage_missing)


# Bar chart to display missing values
plt.figure(figsize=(10, 6))
sns.barplot(x=missing_values.index, y=missing_values.values)
plt.xticks(rotation=45)
plt.title('Valores Faltantes por Columna')
plt.show()

# Class imbalance analysis
class_distribution = df['Potability'].value_counts()
print(class_distribution)

# Distribution of the target variable
sns.countplot(x='Potability', data=df)
plt.show()

# Feature Selection
X = df.drop('Potability', axis=1)
y = df['Potability']
selector = SelectKBest(score_func=f_classif, k='all')
X_new = selector.fit_transform(X, y)

# Print selected features and ratings
features = X.columns[selector.get_support()]
feature_scores = selector.scores_
print("Características Seleccionadas:\n", features)
print("Puntuaciones de Características:\n", feature_scores)

# Viewing Feature Distribution
for feature in X.columns:
    sns.kdeplot(data=df, x=feature, hue='Potability', fill=True, common_norm=False)
    plt.title(f'Distribución de {feature} por Potability')
    plt.show()


# With Pairplot
sns.pairplot(df, hue='Potability', diag_kind='kde')
plt.show()

# Correlation matrix
correlation_matrix = df.corr()

# Improved correlation matrix display
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.xticks(rotation=45)
plt.title('Matriz de Correlación')
plt.show()


df.hist(bins=20, figsize=(15, 10))
plt.show()

plt.figure(figsize=(15, 10))
sns.boxplot(data=df, orient='h')
plt.show()