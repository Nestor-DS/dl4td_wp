# pip install imbalanced-learn

import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

def clean_data(data):
    df = data.copy()
    
    X = df.drop(columns=['Potability'])  
    y = df['Potability']             
    
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                              pd.Series(y_resampled, name='Potability')], axis=1)
    
    imputer = IterativeImputer()
    df_imputed = pd.DataFrame(imputer.fit_transform(df_resampled), columns=df_resampled.columns)

    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    clf = IsolationForest(random_state=0)
    outliers = clf.fit_predict(df_imputed.select_dtypes(include=['float64', 'int64']))
    df_cleaned = df_imputed[outliers == 1]

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_cleaned.select_dtypes(include=['float64', 'int64'])), 
                             columns=df_cleaned.select_dtypes(include=['float64', 'int64']).columns)
    
    return df_scaled
