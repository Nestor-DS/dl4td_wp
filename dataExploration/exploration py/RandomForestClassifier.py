import pandas as pd  # For handling data in the form of dataframes
import numpy as np  # For numerical operations and mathematical calculations
import matplotlib.pyplot as plt  # For data visualization

import seaborn as sns  # For advanced data visualization

from sklearn.preprocessing import StandardScaler  # For data normalization
from sklearn.ensemble import RandomForestClassifier  # For the RandomForest classification model
from sklearn.model_selection import train_test_split  # For splitting the data into training and testing sets
from sklearn.metrics import classification_report, confusion_matrix  # For evaluating the model performance
from sklearn.model_selection import GridSearchCV  # For hyperparameter searching in the model

# Read in the data
df = pd.read_csv('drinking_water_potability.csv')

# Inicializa el escalador
scaler = StandardScaler()

df_normalized = df.copy()
df_normalized[df_normalized.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

# Boxplot para identificar valores atípicos
plt.figure(figsize=(15, 10))
sns.boxplot(data=df_normalized, orient='h')
plt.show()

# Cálculo del rango intercuartílico (IQR)
Q1 = df_normalized.quantile(0.25)
Q3 = df_normalized.quantile(0.75)
IQR = Q3 - Q1

# Identificación de valores atípicos basados en IQR
outliers = ((df_normalized < (Q1 - 1.5 * IQR)) | (df_normalized > (Q3 + 1.5 * IQR))).any(axis=1)

# Verifica si hay variables categóricas (si las hay, debes aplicar codificación)
categorical_columns = df.select_dtypes(include=['object']).columns

if not categorical_columns.empty:
    # Aplica codificación (por ejemplo, one-hot encoding)
    df_encoded = pd.get_dummies(df, columns=categorical_columns)
else:
    df_encoded = df.copy()


# Inicializa el modelo (puedes elegir otro modelo según tus necesidades)
model = RandomForestClassifier(random_state=42)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    df_normalized.drop('Potability', axis=1), 
    df_normalized['Potability'], test_size=0.2, random_state=42)

# Entrenamiento del modelo
model.fit(X_train, y_train)

# Predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Definición de hiperparámetros para ajustar
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}

# Inicialización del modelo con búsqueda de cuadrícula (GridSearchCV)
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Mejores hiperparámetros encontrados
best_params = grid_search.best_params_