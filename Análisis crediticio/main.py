import pandas as pd
from src import config
from src.data_processing import delete_nulls, delete_outliers, data_processing_pipeline
from src.modeling import pipeline_training
from src.utils import save_model
from src.eda import plot_correlation, categorical_correlation
import os

if __name__ == "__main__":
    # Cambia el directorio de trabajo al directorio donde está main.py
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(config.RAW_DATA_PATH)
    
    # Esto es para visualizar gráficas
    df_graphs = delete_nulls(df)
    df_graphs = delete_outliers(df_graphs)
    
    X = df_graphs.drop(columns=config.TARGET_COLUMN)
    y = df_graphs[config.TARGET_COLUMN]

    X_processed = data_processing_pipeline().fit_transform(X)
    df_graphs = X_processed.copy()
    df_graphs[config.TARGET_COLUMN] = y.values

    plot_correlation(df_graphs)
    categorical_correlation(df_graphs)
    
    model = pipeline_training(df)
    save_model(model, "models/xgboost.pkl")