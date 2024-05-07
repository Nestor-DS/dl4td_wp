# pip install imbalanced-learn

import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

def clean_data(data):
    """
    Limpia y preprocesa los datos, aplicando sobremuestreo con SMOTE para equilibrar las clases.
    
    Args:
        data (pd.DataFrame): DataFrame con los datos a limpiar.
    
    Returns:
        df_scaled (pd.DataFrame): DataFrame con los datos limpios, normalizados y clases equilibradas.
    """
    # Copia de los datos originales para preservarlos
    df = data.copy()
    
    # Aplicar sobremuestreo con SMOTE para equilibrar las clases
    X = df.drop(columns=['target_column'])  # Ajusta 'target_column' al nombre de tu columna objetivo
    y = df['target_column']                  # Ajusta 'target_column' al nombre de tu columna objetivo
    
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Reconstruir el DataFrame después del sobremuestreo
    df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                              pd.Series(y_resampled, name='target_column')], axis=1)
    
    # Imputación de valores faltantes
    imputer = IterativeImputer()
    df_imputed = pd.DataFrame(imputer.fit_transform(df_resampled), columns=df_resampled.columns)

    # Detección y eliminación de valores atípicos
    clf = IsolationForest(random_state=0)
    outliers = clf.fit_predict(df_imputed.select_dtypes(include=['float64', 'int64']))
    df_cleaned = df_imputed[outliers == 1]

    # Normalización de los datos
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_cleaned.select_dtypes(include=['float64', 'int64'])), 
                             columns=df_cleaned.select_dtypes(include=['float64', 'int64']).columns)
    
    return df_scaled