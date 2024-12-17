import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')  # Cambiar el backend a TkAgg
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./data/archivo_limpio.csv')# Importar librerías necesarias

# Filtrar las columnas que necesitamos para el clustering
df_clustering = df[['genre', 'duration_ms', 'popularity']]

# Codificar la columna 'genre' de forma numérica, ya que KMeans no puede trabajar con cadenas de texto
df_clustering.loc[:, 'genre'] = df_clustering['genre'].astype('category').cat.codes

# Escalar las características para que todas estén en la misma escala
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_clustering)

# Aplicar KMeans para realizar clustering, probemos con 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(df_scaled)

# Añadir la columna de clusters al DataFrame original
df['cluster'] = df['cluster'].astype('category')

# Ver los primeros 5 registros con su asignación de cluster
print(df[['artist', 'song', 'genre', 'duration_ms', 'popularity', 'cluster']].head())

# Visualización de los clusters
plt.figure(figsize=(10, 6))

# Haremos un gráfico de dispersión para ver cómo se agrupan los géneros, duración y popularidad
sns.scatterplot(data=df, x='duration_ms', y='popularity', hue='cluster', palette='tab10', style='genre', s=100, alpha=0.7)
plt.title('Clustering de canciones por duración y popularidad')
plt.xlabel('Duración (ms)')
plt.ylabel('Popularidad')
plt.legend()
plt.show()
