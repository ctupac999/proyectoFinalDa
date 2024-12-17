import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Cargar el DataFrame limpio (ya procesado)
@st.cache_data  # Para mejorar la eficiencia del dashboard
def load_data():
    # Cargar el DataFrame limpio que ya has guardado
    df = pd.read_csv('./data/archivo_limpio.csv')  # Asegúrate de reemplazar con la ruta correcta
    return df

df = load_data()

# Título del dashboard
st.title("Dashboard de Música")

# Estadísticas descriptivas
st.subheader("Estadísticas Descriptivas")
st.write(df.describe())

# Filtros interactivos
st.sidebar.header("Filtros")

# Mostrar el año más reciente por defecto en el selectbox
selected_year = st.sidebar.selectbox(
    "Seleccionar Año", 
    sorted(df['year'].unique(), reverse=True),  # Ordenamos de manera descendente
    index=0  # El primer elemento será el año más reciente
)

selected_genre = st.sidebar.selectbox("Seleccionar Género", df['genre'].unique())

# Filtrar los datos según la selección
filtered_df = df[(df['year'] == selected_year) & (df['genre'] == selected_genre)]

# Verifica si hay suficientes datos para aplicar KMeans
if len(filtered_df) >= 3:  # Al menos 3 muestras para 3 clusters
    # Selección de las características que usarás para el clustering (asegurarte de elegir solo columnas numéricas)
    X = filtered_df[['duration_ms', 'popularity']]  # Ejemplo: selecciona las columnas relevantes

    # Inicializar el modelo KMeans con 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42)
    
    # Ajustar el modelo y predecir los clusters
    filtered_df['cluster'] = kmeans.fit_predict(X)
    
    # Mostrar el DataFrame con la asignación de clusters
    st.subheader("Clusters Asignados")
    st.dataframe(filtered_df)
else:
    st.warning("No hay suficientes datos para realizar el clustering.")

# Mostrar los primeros registros del dataset filtrado
st.subheader(f"Primeras filas del dataset filtrado ({selected_genre}, {selected_year})")
st.dataframe(filtered_df)

# Gráfico de popularidad por artista
st.subheader(f"Popularidad por Artista en {selected_year} - {selected_genre}")
top_artists = filtered_df.groupby('artist')['popularity'].mean().sort_values(ascending=False).head(10)
fig, ax = plt.subplots()
top_artists.plot(kind='barh', ax=ax, color='skyblue')
ax.set_xlabel('Popularidad Promedio')
ax.set_title(f"Top 10 Artistas de {selected_genre} en {selected_year}")
st.pyplot(fig)

# Gráfico de duración de canciones por artista
st.subheader(f"Duración Promedio de Canciones por Artista en {selected_year} - {selected_genre}")
avg_duration = filtered_df.groupby('artist')['duration_ms'].mean().sort_values(ascending=False).head(10)
fig, ax = plt.subplots()
avg_duration.plot(kind='barh', ax=ax, color='lightgreen')
ax.set_xlabel('Duración Promedio (ms)')
ax.set_title(f"Top 10 Artistas con Mayor Duración Promedio de Canciones")
st.pyplot(fig)

# Análisis de Popularidad vs Duración
st.subheader(f"Relación entre Popularidad y Duración en {selected_year} - {selected_genre}")
fig, ax = plt.subplots()
sns.scatterplot(data=filtered_df, x='duration_ms', y='popularity', hue='artist', ax=ax, palette='Set1')
ax.set_title(f"Popularidad vs Duración de Canciones - {selected_genre} ({selected_year})")
ax.set_xlabel('Duración (ms)')
ax.set_ylabel('Popularidad')
st.pyplot(fig)

# Descargar los datos filtrados como CSV
st.download_button(
    label="Descargar Datos Filtrados",
    data=filtered_df.to_csv(index=False),
    file_name=f"datos_filtrados_{selected_genre}_{selected_year}.csv",
    mime="text/csv"
)

# Tendencias de popularidad por género
genre_trends = df.groupby(['year', 'genre'])['popularity'].mean().reset_index()

# Gráfico de tendencias de popularidad por género
st.subheader("Tendencias de Géneros Populares a lo Largo del Tiempo")
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x='year', y='popularity', hue='genre', data=genre_trends, ax=ax, marker='o')
ax.set_title('Tendencias de Géneros Populares a lo Largo del Tiempo')
ax.set_xlabel('Año')
ax.set_ylabel('Popularidad Promedio')
st.pyplot(fig)

# Gráfico adicional: Evolución de la popularidad de los artistas más populares a lo largo del tiempo
st.subheader(f"Evolución de la Popularidad de los 10 Artistas Más Populares")
top_artists_evolution = df[df['artist'].isin(top_artists.index)]
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=top_artists_evolution, x='year', y='popularity', hue='artist', ax=ax)
ax.set_title('Evolución de la Popularidad de los Artistas Más Populares')
ax.set_xlabel('Año')
ax.set_ylabel('Popularidad Promedio')
st.pyplot(fig)

# Gráfico adicional: Duración promedio de las canciones por género
st.subheader('Duración Promedio de Canciones por Género')
duration_by_genre = df.groupby('genre')['duration_ms'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
duration_by_genre.plot(kind='bar', ax=ax)
ax.set_title('Duración Promedio de Canciones por Género')
ax.set_xlabel('Género')
ax.set_ylabel('Duración Promedio (ms)')
st.pyplot(fig)

# Visualización para consultoras: Oportunidades de festivales según la popularidad de géneros
st.title('Dashboard para Consultoras de Música y Festivales')

# Mostrar las tendencias de los géneros
st.subheader('Tendencias de Géneros Populares a lo Largo del Tiempo')
st.pyplot(fig)

# Mostrar la duración por género
st.subheader('Duración Promedio de Canciones por Género')
st.write(duration_by_genre)
st.pyplot(fig)
