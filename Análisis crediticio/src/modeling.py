import seaborn as sns
import matplotlib.pyplot as plt
import optuna as op
from . import config
from .data_processing import data_processing_pipeline, delete_nulls, delete_outliers
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def train_test_split_data(df):
    X = df.drop(columns=config.TARGET_COLUMN)
    y = df[config.TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test

def build_pipeline(params=None):
    """Crea un pipeline con SMOTE y XGBoost"""
    steps = [
        ('data_processing', data_processing_pipeline()),
        ('smote', SMOTE(random_state=42)),
        ('model', XGBClassifier(**(params or {})))
    ]
    return Pipeline(steps)

def objective(trial, X_train, y_train):
    params = {
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.5),
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'gamma': trial.suggest_float('gamma', 0.3, 0.8)
        }

    pipeline = build_pipeline(params)
    
    return cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy').mean()

def train_model(X_train, y_train, X_test, y_test):
    study = op.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=20)
    best_params = study.best_params
    
    pipeline = build_pipeline(best_params)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    return pipeline, X_train, y_train, X_test, y_test, y_pred

def confusion_matrix_plot(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
    xticklabels=["No concedido préstamo", "Concedido préstamo"], yticklabels=["No concedido préstamo", "Concedido préstamo"])
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.savefig('reports/confusion_matrix.png', bbox_inches='tight')
    #plt.show()
    
def classification_report_plot_and_scores(model, X_train, y_train, X_test, y_test, y_pred):
    print(classification_report(y_test.astype(int), y_pred))
    print('score_test = ', model.score(X_test, y_test.astype(int)))
    print('score_train = ', model.score(X_train, y_train.astype(int)))
    
def pipeline_training(df):

    df = delete_nulls(df)
    df = delete_outliers(df)

    X_train, y_train, X_test, y_test = train_test_split_data(df)
    
    model_and_data = train_model(X_train, y_train, X_test, y_test)
    
    confusion_matrix_plot(*model_and_data[4:6])
    
    classification_report_plot_and_scores(*model_and_data)
    
    return model_and_data[0]  # Retorna el modelo entrenado