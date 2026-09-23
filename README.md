# 🚕 NYC Traffic & Mobility Analytics by Taxómanos

[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> *Extracción, transformación, análisis y predicción de datos de taxis amarillos, los servicios VTC (Uber/Lyft), el tráfico vehicular, eventos y clima en la ciudad de Nueva York durante 2023.*)

---

##  Índice
1. [Descripción del Proyecto](#descripcion)
2. [Orígenes de Datos](#origenes)
3. [Estructura del Proyecto](#estructura)
4. [Instalación y Requisitos](#instalacion)
5. [Variables de Entorno (MinIO y rutas)](#variables)
6. [Uso y Ejecución](#uso)
7. [Trabajo realizado](#trabajo)
8. [Autores (Grupo Taxómanos)](#autores)

---

<a id="descripcion"></a>
##  Descripción del Proyecto

**Objetivos principales:**
* Objetivo 1: Explorar y extraer datos útiles sobre taxis, ubers, clima, eventos y tráfico en NYC.
* Objetivo 2: Limpiar, transformar y unificar datasets masivos usando formato Parquet.
* Objetivo 3: Analizar patrones temporales de movilidad.
* Objetivo 4: Desarrollar gráficos interactivos en HTML para la extracción de conclusiones (zonas con más afluencia, horas con más afluencia, motivos de ello...).
* Objetivo 5: Crear modelos de predicción sobre demanda e ingresos para los taxistas en NYC.
* Objetivo 6: Despliegue de modelos, gráficas y mapas en una página web.

---

<a id="origenes"></a>
## Orígenes de Datos

Los datos utilizados en este proyecto provienen de fuentes públicas y han sido procesados para su análisis:

* **Yellow Taxi Trip Data (YLC)**: Este Dataset proviene del catálogo de la Comisión de Taxis y Limusinas (TLC) del Ayuntamiento de Nueva York (publicado en NYC OpenData) y contiene registros de los viajes realizados en Taxis Amarillos a lo largo del año 2023 en la ciudad de Nueva York. Cada fila del dataset corresponde a un solo viaje en taxi, e incluye fechas y horas de inicio y final de viaje, ID de las zonas de inicio y final según el sistema de Taxi Zones, número de pasajeros, distancia recorrida, coste del viaje, tarifas adicionales, propinas y forma en la que se realizó el pago.

* **Vehículos de Alquiler (Uber/Lyft - FHV):** (High-Volume For-Hire Vehicle (FHV) Trip Data 2023) Este dataset de la mencionada TLC contiene registros detallados de viajes realizados por vehículos de alquiler de alto volumen durante el año 2023. Se consideran FHV de alto volumen aquellas compañías que superan los 10.000 viajes mensuales (como Uber o Lyft). Cada registro corresponde a un viaje individual y contiene variables temporales, espaciales y económicas.. En particular, incluye la fecha y hora de recogida y finalización, los identificadores de la zona de origen y destino según el sistema oficial de “Taxi Zones” de NYC, la distancia recorrida en millas y la duración del trayecto, así como distintos componentes tarifarios (tarifa base abonada por el pasajero, los peajes, las propinas y pago al conductor).

* **Tráfico de NYC:** El dataset es Traffic Volume Counts (Historical) y viene de la web NYC OpenData. Este es un conjunto de datos histórico estático. El Departamento de Transporte de la Ciudad de Nueva York (NYC DOT) utiliza Registradores Automáticos de Tráfico (ATR) para recopilar recuentos de muestras del volumen de tráfico en cruces de puentes y carreteras.

* **Eventos en NYC:** El dataset es NYC Permitted Event Information - Historical y viene de la web NYC OpenData. Se trata de un conjunto de datos histórico estático que recoge información sobre eventos que requieren permiso oficial en la ciudad de Nueva York, como desfiles, festivales, carreras, eventos culturales, rodajes o concentraciones públicas. Cada registro incluye información como el nombre del evento, tipo de evento, agencia responsable, distrito (borough), localización, y fechas y horas de inicio y finalización. Este dataset permite identificar eventos multitudinarios que potencialmente pueden alterar los patrones normales de tráfico y movilidad en la ciudad.

* **Partidos de la MLB en NYC:** Utilizando la MLB Stats API, la API oficial de datos de Major League Baseball (MLB), hemos obtenido datos sobre partidos de béisbol en los dos estadios de los equipos más importantes de Nueva York, Yankee Stadium y Citi Field. Hemos utilizado estos datos para complementar los obtenidos de eventos, ya que estos partidos son los que más gente mueven y pueden alterar el tráfico significativamente.

* **Datos meteorológicos en NYC:** Para incorporar condiciones meteorológicas, se utilizó la API de Open-Meteo, un servicio gratuito que proporciona datos históricos. Se descargaron variables de temperatura, precipitaciones y niveles de nieve en cada hora para la ciudad de Nueva York. Estos datos permiten controlar el efecto del clima sobre el volumen de tráfico.

* **Restaurantes en NYC**: Dataset de restaurantes de Nueva York basado en información de Google Maps, obtenido desde Kaggle. Incluye nombre, rating, cantidad de reseñas, categoría de precio, dirección, coordenadas y código postal. Se utiliza como capa socioeconómica complementaria para analizar distribución de oferta gastronómica y su relación con zonas de demanda.

* **Precios de Alquiler en NYC**: Dataset ligero de propiedades en alquiler de Nueva York, obtenido desde Kaggle. Incluye barrio, latitud, longitud, tipo de habitación, precio y variables de ocupación/reseñas. Este dataset se agrega espacialmente por Taxi Zone para construir indicadores de alquiler medio y mediano como proxy socioeconómico adicional.

---

<a id="estructura"></a>
## Estructura del Proyecto

El repositorio está organizado de forma modular, separando claramente los datos crudos, los datos procesados y el código fuente:

```text
pd2/
├── .env                           # Variables de entorno locales (no commitear)
├── .python-version                # Version de Python para el proyecto
├── .gitignore                     # Archivos que no se suben a GitHub
├── uv.lock                        # Lockfile de uv
├── pyproject.toml                 # Dependencias y metadatos
├── requirements.txt               # Dependencias (compatibilidad pip)
├── borrar_modelos_minio.py        # Limpieza de modelos en MinIO
├── datos/
│   ├── crudos/                    # Datos originales
│   ├── limpios/                   # Datos procesados (parquet, etc.)
│   └── salidas_html/              # Mapas/outputs HTML
├── outputs/
│   └── modelos/
│       └── baseline/              # Modelos y metricas base
├── taxi_zones/                    # Shapefiles de Taxi Zones
├── temp/                          # Temporales de Spark
└── src/
   ├── 01_Extraccion/             # Descarga de fuentes (TLC, clima, eventos, etc.)
   ├── 02_Transformacion/         # Limpieza, parquet y agregaciones
   ├── 03_Modelos/                # Modelos ML, MinIO, patrones, predicciones
   ├── 04_Visualizaciones/        # Dashboards y mapas interactivos
   ├── 05_extraccion2/            # Extraccion fase 2
   ├── 06_agregacion2/            # Cruces y agregaciones fase 2
   ├── 07_score/                  # Scoring y validaciones fase 2
   ├── 08_despliegue/             # HTML finales para despliegue
   └── Visualizacion/             # Visualizaciones modelos
```

---

<a id="instalacion"></a>
##  Instalación y Requisitos

Para replicar este proyecto en tu máquina local, sigue estos pasos:

Requisitos base: Python >= 3.11.

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/bautipelossi/pd2.git
   cd pd2
   ```

2. **Instala dependencias con uv (recomendado):**
   ```bash
   uv venv
   uv sync
   ```

3. **Alternativa con pip:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # En Windows
   pip install -r requirements.txt
   ```

<a id="variables"></a>
## Variables de Entorno (MinIO y rutas)

Crea un archivo .env en la raiz del proyecto (no commitear) con las variables que usa la capa de MinIO y los modelos:

```bash
MINIO_ENDPOINT=https://minio.fdi.ucm.es
MINIO_ACCESS_KEY=TU_ACCESS_KEY
MINIO_SECRET_KEY=TU_SECRET_KEY
MINIO_BUCKET=pd2
MINIO_GROUP_PATH=taxomanos
MINIO_PATH_STYLE=true
MINIO_TAXI_PATH=s3a://pd2/taxomanos/limpios/nyc_taxi_clean.parquet
MINIO_FHV_PATH=s3a://pd2/taxomanos/limpios/fhv_2023_clean.parquet

# Backups locales opcionales
LOCAL_TAXI_PATH=datos/limpios/nyc_taxi_clean.parquet
LOCAL_FHV_PATH=datos/limpios/fhv_2023_clean.parquet
RESTAURANTS_CSV_PATH=datos/crudos/restaurantes_nyc_clean.csv
```

---

<a id="uso"></a>
##  Uso y Ejecución

Para replicar el analisis, sigue el flujo del pipeline. El orden base es:
**1. Extraccion ➔ 2. Transformacion ➔ 3. Modelos/Scoring ➔ 4. Visualizaciones/Despliegue**

### Paso 1: Extracción de Datos
Primero, ejecuta los scripts de la carpeta `src/01_Extraccion` para obtener los datos crudos (clima, eventos, viajes de FHV/LTC, etc.):

```bash
python src/01_Extraccion/FHV.py
python src/01_Extraccion/LTC.py
# (Ejecutar el resto de scripts según los datos que necesites actualizar)
```

### Paso 2: Transformación y Limpieza

Una vez tengas los datos originales, ejecuta los scripts de la carpeta `src/02_Transformacion`. Estos scripts limpiaran los datos, unificaran formatos y generaran los archivos `.parquet` optimizados y listos para el analisis:

```bash
python src/02_Transformacion/Cleaning_FHV.py
python src/02_Transformacion/Cleaning_LTC.py
python src/02_Transformacion/PreprocesamientoVolumenTrafico.py
# (Continúa con los demás scripts de limpieza correspondientes)
```

### Paso 3: Modelos (Fase 1)

Entrenamiento y evaluacion de modelos base sobre datasets limpios para predicciones (propinas, demanda maxima) y patrones. Esta fase genera artefactos en `outputs/` y puede apoyarse en MinIO si esta configurado.

Entrena o valida modelos segun el objetivo:

```bash
python src/03_Modelos/baseline_1a.py
python src/03_Modelos/prediccion_propinas.py
python src/03_Modelos/prediccion_maxima_demanda.py
python src/03_Modelos/identificacion_patrones.py
python src/03_Modelos/prediccion_propinas.py
python src/03_Modelos/nycgeocore.py
```

### Paso 4: Visualizaciones (Fase 1)

Con los datos procesados, finalmente puedes ejecutar los scripts de la carpeta `Visualizacion` para generar los gráficos interactivos. Tienes varios scripts dependiendo del análisis que quieras realizar:

* Para analisis general de demanda cruzada con el clima:
  ```bash
   python src/Visualizacion/YLC_vs_FHV_clima.py
  ```
  (Los resultados de visualización general se exportarán a la subcarpeta Mapa_Interactivo_FHV_TLC/)

* Para la comparativa cruzada de demanda vs. trafico vehicular:
  ```bash
   python src/Visualizacion/visualizacion_agregaciones_con_trafico.py
  ```
  (Los resultados y gráficos .html interactivos se guardarán automáticamente en la subcarpeta Reporte_Trafico_NYC/)

Además, en `src/04_Visualizaciones` hay dashboards y simuladores interactivos listos para ejecutar.

### Paso 5: Fase 2 (extraccion, agregacion, scoring y despliegue web)

Esta fase incluye extraccion 2, agregaciones 2, scoring y la visualizacion final para despliegue en producción.

```bash
python src/05_extraccion2/generar_target_fase2.py
python src/06_agregacion2/cruce_fase2.py
python src/07_score/score_rf.py
```

Los HTML finales estan en `src/08_despliegue` (incluye mapas diarios).


---


<a id="trabajo"></a>
## Trabajo realizado

* Integracion de multiples fuentes (TLC, clima, eventos, trafico, restaurantes y alquileres).
* Limpieza y normalizacion de datasets masivos y conversion a Parquet.
* Agregaciones temporales y espaciales para analisis de demanda.
* Fase 1: modelado base y modelos de prediccion (propinas, maxima demanda) con artefactos en `outputs/`.
* Fase 2: pipeline de extraccion, cruce, scoring y validacion con resultados listos para despliegue.
* Despliegue en producción: dashboards y HTML en `src/08_despliegue` y `src/04_Visualizaciones`.
* Ejercicios adicionales que ayudaron al desarrollo del modelo final: Modelo de zona máxima de demanda, predicción de propinas, identificación de patrones, mapa coroplético del poder adquisitivo.

---

<a id="autores"></a>
## Autores (Grupo Taxómanos)

* **Pablo Alonso** - *Data Scientist / Developer*
* **Rodrigo Jesús-Portanet** - *Data Scientist / Developer*
* **Bautista Pelossi Schweizer** - *Data Scientist / Developer*
* **Óscar Marín** - *Data Scientist / Developer*
