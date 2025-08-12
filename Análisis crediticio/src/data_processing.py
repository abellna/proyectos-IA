import pandas as pd
from sklearn.calibration import LabelEncoder
from . import config
from src.eda import plot_correlation, graficar_correlacion_categorica

def load_data(filepath):
    """Carga datos desde un CSV."""
    return pd.read_csv(filepath)

def initial_cleaning(df):
    """Conversión de tipos y limpieza básica."""
    df = df.astype({'age': 'int64'})
    return df

def delete_nulls(df):
    """Elimina filas con valores nulos."""
    return df.dropna(subset=['purpose'])

def delete_outliers(df):
    """Elimina filas con valores atípicos en edad, experiencia y puntaje de riesgo."""
    df = df[df['age'] <= 100]
    df = df[df['yrs_exp'] <= 60]
    df = df[(df['risk_score'] <= 10000) & (df['risk_score'] >= 0)]
    return df

def fix_categorical_values (df):
    """Corrige valores categóricos específicos."""
    df['gender'] = df['gender'].replace('femal','female')
    df['ownership'] = df['ownership'].str.strip()
    return df

def encoder_values (df):
    """Codifica variables categóricas."""
    le = LabelEncoder()
    categoriasLabelEncoder = ['education', 'purpose']
    categoriasDummies = ['gender', 'ownership', 'previous_loans']

    for i in categoriasDummies:
        df = pd.get_dummies(df, columns=[i], prefix=i, dtype=int)

    df.drop('gender_female', axis=1, inplace=True)
    df.drop('previous_loans_No', axis=1, inplace=True)

    for i in categoriasLabelEncoder:
        df[i] = le.fit_transform(df[i])
        
    return df

def select_columns(df):
    """Selecciona columnas específicas del DataFrame que mejor puntuación en el mapa de calor ha tenido."""
    columns_df = config.SELECTED_COLUMNS + [config.TARGET_COLUMN]
    return df[columns_df]

def data_processing_pipeline(filepath, visualization=True, save_processed=True, overwrite=False):
    """Pipeline de procesamiento de datos."""
    
    if config.PROCESSED_PATH.exists() and not overwrite:
        return pd.read_csv(config.PROCESSED_PATH)
    
    df = load_data(filepath)
    df = initial_cleaning(df)
    df = delete_nulls(df)
    df = delete_outliers(df)
    df = fix_categorical_values(df)
    df = encoder_values(df)
    if visualization:
        plot_correlation(df)
        graficar_correlacion_categorica(df)
    df = select_columns(df)
    
    if save_processed:
        config.PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.PROCESSED_PATH, index=False)
    
    return df