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


#==============================================================================
# IMPUTACIÓN DE VALORES FALTANTES USADO LA MEDIANA
#==============================================================================

from sklearn.impute import SimpleImputer

# Función para realizar la imputación de valores faltantes
def impute_missing_values(file_path):
    # Cargar el archivo Excel en un DataFrame de Pandas
    df = pd.read_csv(file_path)

    # Identificar columnas numéricas y categóricas
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # Imputación para columnas numéricas (usando la mediana)
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Imputación para columnas categóricas (usando la categoría más frecuente)
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return df


#==============================================================================
# ELIMINAR COLUMNAS CON UN FALTANTE DE MÁS DEL 35% DE LOS DATOS
#===============================================================================

def remove_columns_with_missing_values(df, threshold):
    # Calcular el porcentaje de valores faltantes en cada columna
    missing_percent = df.isnull().mean()

    # Seleccionar las columnas que no superen el umbral especificado
    cols_to_keep = missing_percent[missing_percent < threshold].index

    # Eliminar las columnas que superen el umbral especificado
    df = df[cols_to_keep]

    return df

#==============================================================================
# LIMPIEZA DE ARCHIVOS USANDO LA MEDIA
#==============================================================================

def remove_nulls_mean(df):
    # Reemplazar los valores nulos con la media de cada columna
    cleaned_df = df.fillna(df.mean())
    return cleaned_df
