import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Cargar el dataset
data = pd.read_csv("C:/Users/nesto/gitProjects/dl4td_wp/data/drinking_water_potability.csv")

# Preprocesar los datos
X = data.drop('Potability', axis=1)
y = data['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluar el modelo
predictions = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, predictions)}')

# Guardar el modelo como PKL
with open("C:/Users/nesto/gitProjects/dl4td_wp/model.pkl", "wb") as file:
    pickle.dump(model, file)
