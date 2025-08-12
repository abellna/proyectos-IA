import seaborn as sns
import matplotlib.pyplot as plt
import optuna as op
from . import config
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

def resample_data(X_train, y_train, X_test, y_test):
    smote = SMOTE(random_state=42)

    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    return X_train_resampled, y_train_resampled, X_test, y_test
    
def train_model(X_train, y_train, X_test, y_test):
    study = best_params(X_train, y_train)
    
    model = XGBClassifier(max_depth=study.best_params['max_depth'], learning_rate=study.best_params['learning_rate'], n_estimators=study.best_params['n_estimators'], gamma=study.best_params['gamma'])
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    return model, X_train, y_train, X_test, y_test, y_pred

def best_params(X_train, y_train):
    def objective(trial):
        max_depth = trial.suggest_int('max_depth', 5, 15)
        learning_rate = trial.suggest_float('learning_rate', 0.1, 0.5)
        n_estimators = trial.suggest_int('n_estimators', 100, 300)
        gamma = trial.suggest_float('gamma', 0.3, 0.8)

        model = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, gamma=gamma)
        
        return cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()

    study = op.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    
    return study
    

def confusion_matrix_plot(y_test, y_pred):
    conf_matrix = confusion_matrix(y_pred, y_test)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
    xticklabels=["No concedido préstamo", "Concedido préstamo"], yticklabels=["No concedido préstamo", "Concedido préstamo"])
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()
    
    
def classification_report_plot_and_scores(model, X_train, y_train, X_test, y_test, y_pred):
    print(classification_report(y_pred, y_test.astype(int)))
    print('score_test = ', model.score(X_test, y_test.astype(int)))
    print('score_train = ', model.score(X_train, y_train.astype(int)))
    
    
def modeling_pipeline(df):
    """Pipeline de modelado."""
    data_split = train_test_split_data(df)
    resampled_data = resample_data(*data_split)
    model_info = train_model(*resampled_data)
    confusion_matrix_plot(*model_info[4:6])
    classification_report_plot_and_scores(*model_info)
    return model_info[0]  # Return the trained model