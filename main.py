import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import plotly.express as px
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource
from dash import Dash, dcc, html
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

# === 1. Funciones de carga y limpieza ===


def load_data(file_path):
    """Carga los datos desde un archivo CSV."""
    try:
        data = pd.read_csv(file_path)
        print("Archivo cargado correctamente.")
        return data
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        raise


def clean_and_transform_data(data):
    """Limpia y transforma los datos para el análisis."""
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%y')
    data['Year-Month'] = data['Date'].dt.to_period('M')
    data['Weekday'] = data['Date'].dt.day_name()
    data['Hour'] = data['Date'].dt.hour  # Para análisis por hora
    return data

# === 2. Enriquecimiento de datos ===


def categorize_title(title, genre_keywords):
    """Categoriza un título en base a las palabras clave."""
    for genre, keywords in genre_keywords.items():
        if any(keyword.lower() in title.lower() for keyword in keywords):
            return genre
    return 'Anime'


def add_genre_column(data, genre_keywords):
    """Añade una columna de género a los datos."""
    data['Genre'] = data['Title'].apply(
        lambda title: categorize_title(title, genre_keywords))
    return data


def classify_content_type(title):
    """Clasifica si el título es una serie, película o documental."""
    if any(word in title.lower() for word in ['temporada', 'episodio']):
        return 'Serie'
    elif 'documental' in title.lower():
        return 'Documental'
    else:
        return 'Película'


def add_content_type_column(data):
    """Añade una columna de tipo de contenido a los datos."""
    data['Content_Type'] = data['Title'].apply(classify_content_type)
    return data

# === 3. Visualizaciones avanzadas ===


def plot_top_titles(data):
    """Genera un gráfico de barras para los títulos más vistos (Top 5)."""
    top_titles = data['Title'].value_counts().head(5)
    plt.figure(figsize=(12, 8))
    top_titles.plot(kind='bar', color='skyblue')
    plt.title('Top 5 Contenidos Más Vistos', fontsize=14)
    plt.xlabel('Título', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.tight_layout()
    plt.savefig('top_titles.png', dpi=300)
    plt.show()


def plot_weekly_distribution(data):
    """Genera un gráfico de barras para la distribución semanal."""
    weekly_distribution = data['Weekday'].value_counts().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.figure(figsize=(12, 8))
    weekly_distribution.plot(kind='bar', color='lightgreen')
    plt.title('Distribución Semanal de Visualizaciones', fontsize=14)
    plt.xlabel('Día de la Semana', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.tight_layout()
    plt.savefig('weekly_distribution.png', dpi=300)
    plt.show()


def plot_heatmap(data):
    """Genera un gráfico de calor para hábitos de consumo por día y hora."""
    heatmap_data = data.groupby(
        ['Weekday', 'Hour']).size().unstack(fill_value=0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap='coolwarm', annot=False, cbar=True)
    plt.title('Consumo por Día y Hora', fontsize=14)
    plt.xlabel('Hora del Día', fontsize=12)
    plt.ylabel('Día de la Semana', fontsize=12)
    plt.tight_layout()
    plt.savefig('heatmap_consumo.png', dpi=300)
    plt.show()


def plot_pareto(data):
    """Genera un diagrama de Pareto para los títulos más vistos (Top 5)."""
    title_counts = data['Title'].value_counts().head(5)
    cum_percentage = (title_counts.cumsum() / title_counts.sum()) * 100

    plt.figure(figsize=(14, 8))
    ax = title_counts.plot(kind='bar', color='skyblue',
                           label='Frecuencia', width=0.8)
    ax2 = ax.twinx()
    ax2.plot(cum_percentage.index, cum_percentage.values, color='red',
             marker='o', label='Porcentaje acumulado', linestyle='--')

    ax.set_ylabel('Frecuencia', fontsize=12)
    ax2.set_ylabel('Porcentaje acumulado', fontsize=12)
    ax.set_title(
        'Diagrama de Pareto de Contenidos Más Vistos (Top 5)', fontsize=14)
    ax.tick_params(axis='x', labelrotation=45, labelsize=8)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('pareto_contenidos_top5.png', dpi=300)
    plt.show()


def plot_pie_chart(data):
    """Genera un gráfico de pastel para los géneros más vistos (Top 5)."""
    top_genres = data['Genre'].value_counts().head(5)
    plt.figure(figsize=(10, 7))
    top_genres.plot(kind='pie', autopct='%1.1f%%',
                    startangle=140, colors=sns.color_palette('pastel'))
    plt.title('Top 5 Géneros Más Vistos', fontsize=14)
    plt.ylabel('')  # Ocultar la etiqueta del eje y
    plt.tight_layout()
    plt.savefig('pie_chart_top5_genres.png', dpi=300)
    plt.show()


def plot_genre_distribution(data):
    """Genera un gráfico de pastel para la distribución de géneros."""
    genre_counts = data['Genre'].value_counts()
    plt.figure(figsize=(10, 7))
    genre_counts.plot(kind='pie', autopct='%1.1f%%',
                      startangle=140, colors=sns.color_palette('pastel'))
    plt.title('Distribución de Géneros', fontsize=14)
    plt.ylabel('')  # Ocultar la etiqueta del eje y
    plt.tight_layout()
    plt.savefig('genre_distribution.png', dpi=300)
    plt.show()


def plot_interactive_top_titles(data):
    """Genera un gráfico de barras interactivo para los títulos más vistos."""
    top_titles = data['Title'].value_counts().head(10).reset_index()
    top_titles.columns = ['Title', 'Count']

    fig = px.bar(top_titles, x='Title', y='Count', title="Top 10 Contenidos Más Vistos",
                 labels={'Title': 'Título', 'Count': 'Frecuencia'},
                 text='Count', template='plotly_white')

    fig.update_traces(marker_color='salmon', textposition='outside')
    fig.show()


def plot_interactive_heatmap(data):
    """Genera un gráfico de calor interactivo para hábitos de consumo."""
    heatmap_data = data.groupby(
        ['Weekday', 'Hour']).size().unstack(fill_value=0)

    fig = px.imshow(heatmap_data, labels=dict(x="Hora del Día", y="Día de la Semana", color="Frecuencia"),
                    x=heatmap_data.columns, y=heatmap_data.index, color_continuous_scale='Viridis')

    fig.update_layout(
        title="Consumo por Día y Hora (Interactivo)", template='plotly_white')
    fig.show()


def plot_bokeh_top_titles(data):
    """Genera un gráfico de barras interactivo con Bokeh."""
    try:
        output_notebook()
    except ImportError:
        print("El módulo 'IPython' no está instalado. Por favor, instálalo usando 'pip install ipython'.")
        return
    top_titles = data['Title'].value_counts().head(10).reset_index()
    top_titles.columns = ['Title', 'Count']

    source = ColumnDataSource(top_titles)

    p = figure(x_range=top_titles['Title'], height=400, title="Top 10 Contenidos Más Vistos",
               toolbar_location=None, tools="", tooltips="@Title: @Count")

    p.vbar(x='Title', top='Count', width=0.9, source=source,
           color='salmon', legend_field="Title")
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.xaxis.major_label_orientation = 1.2

    show(p)


def plot_bokeh_monthly_distribution(data):
    """Genera un gráfico de líneas interactivo para la distribución mensual."""
    try:
        output_notebook()
    except ImportError:
        print("El módulo 'IPython' no está instalado. Por favor, instálalo usando 'pip install ipython'.")
        return
    monthly_distribution = data['Year-Month'].value_counts().sort_index().reset_index()
    monthly_distribution.columns = ['Month', 'Count']
    source = ColumnDataSource(monthly_distribution)

    p = figure(x_range=monthly_distribution['Month'].astype(str), height=400, title="Visualizaciones Mensuales",
               toolbar_location=None, tools="", tooltips="@Month: @Count")

    p.line(x='Month', y='Count', line_width=2, source=source,
           color="green", legend_label="Frecuencia")
    p.circle(x='Month', y='Count', size=8, source=source,
             color="blue", legend_label="Puntos")

    p.xgrid.grid_line_color = None
    p.xaxis.major_label_orientation = 1.2
    p.legend.location = "top_left"

    show(p)

# === 4. Predicción de géneros favoritos ===


def predict_favorite_genre(data):
    """Predice los géneros favoritos utilizando clustering KMeans."""
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['Title'])

    kmeans = KMeans(n_clusters=3, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X)

    genre_distribution = data.groupby(
        'Cluster')['Genre'].value_counts().unstack(fill_value=0)
    print("Distribución por clústeres y géneros:\n", genre_distribution)
    return genre_distribution

# === 5. Función principal ===


def main():
    # Ruta del archivo
    file_path = os.getenv("NETFLIX_FILE_PATH")

    # Cargar datos
    data = load_data(file_path)

    # Limpiar y transformar datos
    data = clean_and_transform_data(data)

    # Definir géneros y palabras clave asociadas
    genre_keywords = {
        'Acción': ['Jujutsu', 'Avengers', 'Superman', 'Batman', 'Fight'],
        'Drama': ['Emperatriz', 'Crown', 'Tribunal', 'Drama', 'Historia'],
        'Comedia': ['Comedy', 'Funny', 'Risa', 'Sitcom', 'Friends'],
        'Terror': ['Horror', 'Terror', 'Zombie', 'Scary', 'Ghost'],
        'Ciencia Ficción': ['Sci-Fi', 'Alien', 'Space', 'Futuristic', 'Black Mirror'],
        'Animación': ['Anime', 'Cartoon', 'Naruto', 'One Piece', 'Dragon Ball']
    }

    # Añadir columna de género y tipo de contenido
    data = add_genre_column(data, genre_keywords)
    data = add_content_type_column(data)

    # Exploración inicial
    print("\nPrimeras filas de los datos:\n")
    print(data.head())
    print("\nResumen de las columnas:\n")
    print(data.info())

    # Visualizaciones
    plot_top_titles(data)
    plot_weekly_distribution(data)
    plot_heatmap(data)
    plot_pareto(data)
    plot_pie_chart(data)
    plot_interactive_top_titles(data)
    plot_interactive_heatmap(data)
    plot_bokeh_top_titles(data)
    plot_bokeh_monthly_distribution(data)

    # Predicción de géneros favoritos
    genre_clusters = predict_favorite_genre(data)

if __name__ == "__main__":
    main()
