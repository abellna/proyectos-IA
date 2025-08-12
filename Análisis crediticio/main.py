from src.data_processing import data_processing_pipeline
from src.modeling import modeling_pipeline
from src.utils import save_model

if __name__ == "__main__":
    df = data_processing_pipeline("data/raw/loan_data.csv")
    model = modeling_pipeline(df)
    # 4. Guardar modelo
    save_model(model, "models/xgboost.pkl")