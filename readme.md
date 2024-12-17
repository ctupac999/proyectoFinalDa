# Proyecto de Análisis de Música y Tendencias de Géneros

Este proyecto tiene como objetivo analizar tendencias de géneros musicales, popularidad de artistas, duración de las canciones, y otros aspectos relevantes para la industria musical, como las recomendaciones para festivales. Utilizamos un conjunto de datos disponible en Kaggle, que fue limpiado, normalizado, analizado y visualizado para extraer insights que pueden ser útiles para consultoras musicales.

## Estructura del Proyecto

El repositorio está organizado de la siguiente manera:


### Descripción de Archivos 

1. **data/**: Contiene los conjuntos de datos utilizados:
   - `songs_normalize.csv`: El conjunto de datos original, descargado de Kaggle.
   - `archivo_limpio.csv`: El conjunto de datos limpio y procesado, listo para su análisis.

2. **app.py**: Archivo donde se realiza el análisis de datos utilizando el algoritmo KMeans para realizar un clustering de las canciones basado en la duración y popularidad. También contiene algunos análisis adicionales, como la relación entre popularidad y duración de las canciones.

3. **dashboard.py**: Implementación de un dashboard interactivo utilizando Streamlit. Permite a los usuarios filtrar datos por año y género, ver las tendencias de popularidad de los géneros a lo largo del tiempo, la popularidad de los artistas, la duración promedio de las canciones y realizar análisis de KMeans.

4. **main.ipynb**: Notebook de Jupyter que cubre el proceso de limpieza de los datos, exploración, análisis de correlaciones y visualización de las tendencias de los géneros y artistas.

5. **presentacion.pdf**: Presentación de proyecto.

## Descripción del Proyecto

Este proyecto tiene como objetivo analizar la evolución de la música a lo largo del tiempo y explorar diversas características de las canciones, como su popularidad, duración y género. El análisis se realiza a través de la limpieza y procesamiento de datos, visualización de tendencias, y el uso de técnicas de Machine Learning como el algoritmo KMeans para agrupar canciones en clusters según su duración y popularidad.

### Análisis de Datos

El análisis se divide en varias partes:

1. **Limpieza de Datos**:
   - Se procesó y limpió el conjunto de datos original (`songs_normalize.csv`), eliminando valores nulos y corrigiendo formatos incorrectos.
   - Se eliminaron columnas innecesarias y se normalizaron ciertas variables como la duración de las canciones.

2. **Exploración de Datos**:
   - Se exploraron las relaciones entre las diferentes variables, como la correlación entre popularidad y duración de las canciones.
   - Se agruparon los datos por año y género para analizar las tendencias de popularidad.

3. **KMeans**:
   - Se utilizó el algoritmo KMeans para agrupar las canciones en clusters basados en la duración y popularidad. Esto permite identificar patrones y características comunes dentro de las canciones.

4. **Visualización**:
   - Se crearon gráficos de barras, dispersión y líneas para visualizar la evolución de la popularidad de los géneros y artistas a lo largo del tiempo.
   - Se generaron visualizaciones de la duración promedio de las canciones por género.

### Dashboard Interactivo

El dashboard interactivo, desarrollado con **Streamlit**, permite a los usuarios filtrar datos por año y género, ver las tendencias de popularidad por género a lo largo del tiempo, así como la popularidad de los artistas más escuchados y la duración promedio de las canciones. El dashboard también incluye la funcionalidad para descargar los datos filtrados en formato CSV.

#### Funciones del Dashboard:

- **Filtros interactivos**: Los usuarios pueden seleccionar un año y un género para filtrar los datos y ver gráficos actualizados.
- **Gráficos de Popularidad**: Muestra la popularidad promedio de los artistas más populares por género y año.
- **Relación Popularidad vs Duración**: Visualiza la relación entre la duración de las canciones y su popularidad.
- **Clusters de KMeans**: Visualiza el clustering de las canciones basándose en la duración y popularidad.
- **Tendencias de Géneros**: Muestra cómo ha evolucionado la popularidad de los géneros musicales a lo largo del tiempo.
- **Descarga de Datos Filtrados**: Permite descargar los datos filtrados en formato CSV para análisis posteriores.

## Requisitos

- Python 3.8 o superior
- Paquetes necesarios:
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `streamlit`
  - `sklearn`
  - `numpy`
  - `scipy`

### Instalación

Para instalar las dependencias necesarias, puedes crear un entorno virtual e instalar los paquetes usando `pip`:

```bash
# Crear un entorno virtual
python -m venv venv

# Activar el entorno virtual
# En Windows
venv\Scripts\activate
# En macOS/Linux
source venv/bin/activate

# Instalar las dependencias
pip install streamlit pandas matplotlib seaborn scikit-learn
pip install --upgrade numpy pandas
conda update numpy pandas


# Ejecutar dashboard
streamlit run dashboard.py
