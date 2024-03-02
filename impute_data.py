import pandas as pd

def count__nulls(df):
    # Count number of rows
    n_rows = len(df)
    
    # Count number of missing values in each column
    missing_values = df.isnull().sum()
    
    # Create a new dataframe with missing values
    missing_df = pd.DataFrame({
        'column_name': missing_values.index, 
        'missing_values': missing_values.values
    })
    
    return missing_df

def remove__nulls(df):
    # Remove rows with missing values
    df = df.dropna()
    
    return df