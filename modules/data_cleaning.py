import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


def clean_data(data):
    df = data
    
    # Imputation
    
    # Enable the iterative imputer
    imputer = IterativeImputer()

    # Train the imputer on the data with missing values
    imputer.fit(df)

    # Transform the data to impute missing values
    df_imputed = pd.DataFrame(imputer.transform(df), columns=df.columns)

    # Outlier

    # Select only the numeric columns
    numeric_columns = df_imputed.select_dtypes(include=['float64', 'int64'])

    # Initialize and fit the Isolation Forest model
    clf = IsolationForest(random_state=0)
    clf.fit(numeric_columns)

    # Identify outliers (1 for normal values, -1 for outliers)
    outliers = clf.predict(numeric_columns)

    # Filter only the normal (non-outlier) values
    df_cleaned = df_imputed[outliers == 1]
    
    # Normalization
    
    # Select the numeric features
    numeric_columns = df_cleaned.select_dtypes(include=['float64', 'int64'])

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Scale the numeric features
    numeric_columns_scaled = scaler.fit_transform(numeric_columns)

    # Create a DataFrame with scaled features
    df_scaled = pd.DataFrame(numeric_columns_scaled, columns=numeric_columns.columns)
    
    return df_scaled