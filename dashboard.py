import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el DataFrame limpio (ya procesado)
@st.cache_data # Para mejorar la eficiencia del dashboard
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
selected_year = st.sidebar.selectbox("Seleccionar Año", sorted(df['year'].unique()), index=0)
selected_genre = st.sidebar.selectbox("Seleccionar Género", df['genre'].unique())

# Filtrar los datos según la selección
filtered_df = df[(df['year'] == selected_year) & (df['genre'] == selected_genre)]

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


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Cargar el DataFrame (suponiendo que ya lo tienes limpiado y preparado)
# df = pd.read_csv('./data/archivo_limpio.csv')

# Filtrar los datos por género y año
genre_trends = df.groupby(['year', 'genre'])['popularity'].mean().reset_index()

# Crear el gráfico de tendencias de popularidad por género
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x='year', y='popularity', hue='genre', data=genre_trends, ax=ax, marker='o')

ax.set_title('Tendencias de Géneros Populares a lo Largo del Tiempo')
ax.set_xlabel('Año')
ax.set_ylabel('Popularidad Promedio')

st.pyplot(fig)


# artistas mas pupulares

# Filtrar los artistas más populares por año
selected_year = st.selectbox('Selecciona un año:', df['year'].unique())

top_artists_year = df[df['year'] == selected_year].groupby('artist')['popularity'].mean().sort_values(ascending=False).head(10)

# Mostrar los artistas más populares
st.write(f"Los 10 artistas más populares en {selected_year}:")
st.write(top_artists_year)

# Crear un gráfico de barras
fig, ax = plt.subplots(figsize=(10, 6))
top_artists_year.plot(kind='bar', ax=ax)

ax.set_title(f'Artistas Más Populares en {selected_year}')
ax.set_xlabel('Artista')
ax.set_ylabel('Popularidad Promedio')

st.pyplot(fig)


# relacion entre generos y duración ajustada
# Calcular la duración promedio por género
duration_by_genre = df.groupby('genre')['duration_ms'].mean().sort_values(ascending=False)

# Crear un gráfico de barras de duración promedio por género
fig, ax = plt.subplots(figsize=(10, 6))
duration_by_genre.plot(kind='bar', ax=ax)

ax.set_title('Duración Promedio de Canciones por Género')
ax.set_xlabel('Género')
ax.set_ylabel('Duración Promedio (ms)')

st.pyplot(fig)


# visualizacion y resumen para consultoras

# Mostrar un resumen interactivo
st.title('Dashboard para Consultoras de Música y Festivales')

# Gráfico de tendencias de géneros
st.subheader('Tendencias de Géneros Populares a lo Largo del Tiempo')
st.pyplot(fig)  # Gráfico ya generado en el paso 1

# Artistas más populares
st.subheader(f'Los 10 Artistas Más Populares en {selected_year}')
st.write(top_artists_year)  # Los artistas generados en el paso 2
st.pyplot(fig)  # Gráfico de barras de artistas más populares

# Relación entre géneros y duración ajustada
st.subheader('Duración Promedio de Canciones por Género')
st.write(duration_by_genre)  # Duración promedio por género
st.pyplot(fig)  # Gráfico de barras de duración por género
