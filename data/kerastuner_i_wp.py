# Instalación
"""

!pip install keras-tuner
!pip install imbalanced-learn

"""# Code"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from sklearn.experimental import enable_iterative_imputer

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.utils import resample
import pandas as pd



data_path = "C:/Users/nesto/gitProjects/dl4td_wp/data/drinking_water_potability.csv"
df = pd.read_csv(data_path)

df.head(5)



df_use = df

"""# Modeling"""

# Define features and labels
features = df_use.columns[:-1]
labels = df_use.columns[-1]

X_train, X_test, y_train, y_test = train_test_split(df_use[features], df_use[labels], test_size=0.2, random_state=42)

# Initialize StandardScaler
scaler = StandardScaler()

# Scale features
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build model structure. Choose some parameters
def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 20)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model

# Search for best model according to accuracy criterion
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=50,
    executions_per_trial=30,
    directory='my_dir',
    project_name='keras_tuner_1')

# Perform the hyperparameter search
tuner.search(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

best_model = tuner.get_best_models(num_models=1)[0]

y_pred = best_model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype("int32")

accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes)
recall = recall_score(y_test, y_pred_classes)
f1 = f1_score(y_test, y_pred_classes)
auc = roc_auc_score(y_test, y_pred)

# Print the evaluation metrics
print("Evaluation Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print(f"AUC: {auc}")

# Get the summary of results
tuner.results_summary()

# Create a dataframe to store the results
results_dict = {'Model': ['Best Model'],
                'Accuracy': [accuracy],
                'Precision': [precision],
                'Recall': [recall],
                'F1-score': [f1],
                'AUC': [auc]}

results_df = pd.DataFrame(results_dict)

# Display the results dataframe
print(results_df)