import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from . import config

def convert_type(df):
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
    columns_df = config.SELECTED_COLUMNS
    return df[columns_df]

class data_processing_pipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Conversión y limpieza
        df = convert_type(df)

        # Correcciones categóricas
        df = fix_categorical_values(df)

        # Codificación
        df = encoder_values(df)
        
        # Selección de columnas
        df = select_columns(df)

        return df