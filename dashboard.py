import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Cargar el DataFrame limpio (ya procesado)
@st.cache_data  # Para mejorar la eficiencia del dashboard
def load_data():
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
selected_year = st.sidebar.selectbox(
    "Seleccionar Año", 
    sorted(df['year'].unique(), reverse=True),
    index=0  # Año más reciente
)

selected_genre = st.sidebar.selectbox("Seleccionar Género", df['genre'].unique())

# Filtrar los datos según la selección
filtered_df = df[(df['year'] == selected_year) & (df['genre'] == selected_genre)]

# Verifica si hay suficientes datos para aplicar KMeans
if len(filtered_df) >= 3:
    X = filtered_df[['duration_ms', 'popularity']]  # Columnas relevantes
    kmeans = KMeans(n_clusters=3, random_state=42)
    filtered_df['cluster'] = kmeans.fit_predict(X)
    st.subheader("Clusters Asignados")
    st.dataframe(filtered_df)
else:
    st.warning("No hay suficientes datos para realizar el clustering.")

# Mostrar los primeros registros
st.subheader(f"Primeras filas del dataset filtrado ({selected_genre}, {selected_year})")
st.dataframe(filtered_df)

# Gráfico: Popularidad por artista
st.subheader(f"Popularidad por Artista en {selected_year} - {selected_genre}")
top_artists = filtered_df.groupby('artist')['popularity'].mean().sort_values(ascending=False).head(10)
fig, ax = plt.subplots()
top_artists.plot(kind='barh', ax=ax, color='skyblue')
ax.set_xlabel('Popularidad Promedio')
st.pyplot(fig)

# Gráfico: Duración promedio por artista
st.subheader(f"Duración Promedio de Canciones por Artista en {selected_year} - {selected_genre}")
avg_duration = filtered_df.groupby('artist')['duration_ms'].mean().sort_values(ascending=False).head(10)
fig, ax = plt.subplots()
avg_duration.plot(kind='barh', ax=ax, color='lightgreen')
ax.set_xlabel('Duración Promedio (ms)')
st.pyplot(fig)

# Scatterplot: Popularidad vs Duración
st.subheader(f"Relación entre Popularidad y Duración en {selected_year} - {selected_genre}")
fig, ax = plt.subplots()
sns.scatterplot(data=filtered_df, x='duration_ms', y='popularity', hue='artist', ax=ax, palette='Set1')
ax.set_xlabel('Duración (ms)')
ax.set_ylabel('Popularidad')
st.pyplot(fig)

# Descargar datos filtrados
st.download_button(
    label="Descargar Datos Filtrados",
    data=filtered_df.to_csv(index=False),
    file_name=f"datos_filtrados_{selected_genre}_{selected_year}.csv",
    mime="text/csv"
)

# Tendencias de popularidad por género
genre_trends = df.groupby(['year', 'genre'])['popularity'].mean().reset_index()
st.subheader("Tendencias de Géneros Populares a lo Largo del Tiempo")
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x='year', y='popularity', hue='genre', data=genre_trends, ax=ax, marker='o')
st.pyplot(fig)

# Lista de géneros más escuchados por año
st.subheader(f"Géneros Más Escuchados en {selected_year}")
most_popular_genres = (
    df[df['year'] == selected_year]
    .groupby('genre')['popularity']
    .mean()
    .sort_values(ascending=False)
)
st.write(most_popular_genres.reset_index().rename(columns={'genre': 'Género', 'popularity': 'Popularidad Promedio'}))

# Evolución de popularidad de los artistas más populares
st.subheader(f"Evolución de la Popularidad de los 10 Artistas Más Populares")
top_artists_evolution = df[df['artist'].isin(top_artists.index)]
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=top_artists_evolution, x='year', y='popularity', hue='artist', ax=ax)
st.pyplot(fig)

# Duración promedio por género
st.subheader('Duración Promedio de Canciones por Género')
duration_by_genre = df.groupby('genre')['duration_ms'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
duration_by_genre.plot(kind='bar', ax=ax)
ax.set_ylabel('Duración Promedio (ms)')
st.pyplot(fig)

# Visualización para consultoras
st.title('Dashboard para Consultoras de Música y Festivales')
st.subheader('Tendencias de Géneros Populares a lo Largo del Tiempo')
st.pyplot(fig)

st.subheader('Duración Promedio de Canciones por Género')
st.write(duration_by_genre)
