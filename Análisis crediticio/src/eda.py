import seaborn as sns
import matplotlib.pyplot as plt
from . import config

def plot_correlation(df):
    """Muestra el mapa de calor de correlaciones."""
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    # plt.show()
    plt.savefig('reports/correlation_heatmap.png', bbox_inches='tight')

def categorical_correlation(datos):
    columnas = config.SELECTED_COLUMNS
    variable_objetivo = config.TARGET_COLUMN
    
    n = len(columnas)
    filas = (n + 1) // 2 
    fig, ax = plt.subplots(nrows=filas, ncols=2, figsize=(14, 5 * filas))
    ax = ax.flatten()
    
    for i, columna in enumerate(columnas):
        if columna in ['ownership_MORTGAGE', 'ownership_RENT', 'previous_loans_Yes']:
            categoria = datos.groupby(variable_objetivo)[columna].sum()
            categoria.plot(kind='bar', ax=ax[i], color='skyblue', edgecolor='black')
            ax[i].set_title(f"{columna} por {variable_objetivo}")
            ax[i].set_ylabel(columna)
            ax[i].set_xlabel(variable_objetivo)
        if datos[columna].dtype in ['int64', 'float64']: 
            # Creamos un barplot agregando por promedio o conteo
            promedio_por_categoria = datos.groupby(variable_objetivo)[columna].mean()
            promedio_por_categoria.plot(kind='bar', ax=ax[i], color='skyblue', edgecolor='black')
            ax[i].set_title(f"Promedio de {columna} por {variable_objetivo}")
            ax[i].set_ylabel(columna)
            ax[i].set_xlabel(variable_objetivo)
        else:
            sns.countplot(x=columna, hue=variable_objetivo, data=datos, ax=ax[i], palette='viridis')
            ax[i].set_title(f"{columna} distribución por {variable_objetivo}")
    
    # Histograma de la variable objetivo
    if len(columnas) % 2 != 0:  # Si sobra un gráfico, lo usamos para el histograma.
        sns.countplot(x=variable_objetivo, data=datos, ax=ax[-1], palette='muted')
        ax[-1].set_title(f"Distribución de {variable_objetivo}")
    else:  # Si no sobra, creamos un gráfico adicional.
        fig, extra_ax = plt.subplots(1, 1, figsize=(7, 5))
        sns.countplot(x=variable_objetivo, data=datos, ax=extra_ax, palette='muted')
        extra_ax.set_title(f"Distribución de {variable_objetivo}")
    
    plt.tight_layout()
    #plt.show()
    plt.savefig('reports/categorical_correlation_values.png', bbox_inches='tight')