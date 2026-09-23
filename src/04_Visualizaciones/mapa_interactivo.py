import os
import sys
import json
import urllib.request
import platform
import pandas as pd
import plotly.express as px
import streamlit as st
import boto3
from botocore.client import Config
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from pyspark.sql import SparkSession, Row
from pyspark.ml import PipelineModel

# --- CONFIGURACIÓN DE PÁGINA STREAMLIT ---
st.set_page_config(page_title="NYC Taxi Map Simulator", layout="wide", page_icon="🗺️")

def descargar_modelo_desde_minio(ruta_local):
    """Descarga el modelo v2 desde MinIO de forma recursiva si no existe en local"""
    load_dotenv(find_dotenv())
    
    endpoint = os.getenv("MINIO_ENDPOINT")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    bucket = os.getenv("MINIO_BUCKET")
    group_path = os.getenv("MINIO_GROUP_PATH")

    if not all([endpoint, access_key, secret_key, bucket, group_path]):
        st.error("❌ Faltan las variables de entorno de MinIO (.env) para descargar el modelo.")
        st.stop()

    s3_client = boto3.client(
        's3', endpoint_url=endpoint, aws_access_key_id=access_key,
        aws_secret_access_key=secret_key, config=Config(signature_version='s3v4')
    )

    # Buscamos la versión v2 que sabemos que está íntegra en la nube
    prefijo_s3 = f"{group_path}/models/mejor_modelo_demanda_v2/"
    
    texto_estado = st.empty()
    texto_estado.info("📥 El modelo predictivo no está en local. Descargándolo automáticamente de MinIO... (Esto tomará un minuto)")

    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        paginas = paginator.paginate(Bucket=bucket, Prefix=prefijo_s3)
        
        archivos_totales = 0
        for pagina in paginas:
            if 'Contents' in pagina:
                for obj in pagina['Contents']:
                    s3_key = obj['Key']
                    
                    # Ignorar marcadores de carpeta (fantasmas)
                    if s3_key.endswith('/'):
                        continue
                        
                    # Calcular la ruta donde se guardará localmente
                    ruta_relativa = s3_key.replace(prefijo_s3, "")
                    ruta_archivo_local = ruta_local / ruta_relativa
                    
                    # Crear carpetas intermedias si no existen (ej: stages, metadata)
                    ruta_archivo_local.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Descargar archivo
                    s3_client.download_file(bucket, s3_key, str(ruta_archivo_local))
                    archivos_totales += 1
                    
        if archivos_totales == 0:
            texto_estado.error("❌ No se encontró el modelo en MinIO. Avisa al administrador.")
            st.stop()
        else:
            texto_estado.success(f"✅ Modelo sincronizado con éxito desde la nube ({archivos_totales} archivos).")
            
    except Exception as e:
        texto_estado.error(f"❌ Error crítico de red descargando desde MinIO: {e}")
        st.stop()

# --- FUNCIONES DE CACHÉ ---
@st.cache_resource
def iniciar_spark_y_modelo():
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    
    if platform.system() == "Windows":
        os.environ['HADOOP_HOME'] = "C:/hadoop"
    
    spark = SparkSession.builder.appName("NYC_Map_Simulator").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    ruta_modelo = Path(__file__).resolve().parents[1] / "modelos" / "mejor_modelo_demanda"
    
    # --- SISTEMA DE FALLBACK AUTOMÁTICO ---
    # Un modelo de Spark siempre tiene una carpeta 'metadata'. Si no existe, toca descargarlo.
    if not (ruta_modelo / "metadata").exists():
        descargar_modelo_desde_minio(ruta_modelo)
    
    modelo = PipelineModel.load(str(ruta_modelo))
    
    ruta_parquet = str(Path(__file__).resolve().parents[2] / "datos" / "limpios" / "resumen_zona_hora.parquet")
    dataset = spark.read.parquet(ruta_parquet)
    
    columnas_estaticas = ["pulocationid", "num_restaurantes", "precio_medio_rest", "num_alquileres", "precio_medio_alquiler"]
    df_estatico = dataset.select(columnas_estaticas).dropDuplicates(["pulocationid"])
    
    return spark, modelo, df_estatico

@st.cache_data
def obtener_diccionario_zonas():
    url_oficial = "https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv"
    try:
        df_zonas = pd.read_csv(url_oficial)
        return dict(zip(df_zonas['LocationID'], df_zonas['Zone']))
    except:
        return {}

@st.cache_data
def obtener_geometria_mapa():
    url_geojson = "https://data.cityofnewyork.us/api/geospatial/8meu-9t5y?method=export&format=GeoJSON"
    req = urllib.request.Request(url_geojson, headers={'User-Agent': 'Mozilla/5.0'})
    return json.loads(urllib.request.urlopen(req).read())

# --- CÁLCULO DE PREDICCIONES EN TIEMPO REAL ---
def simular_demanda_mapa(spark, modelo, df_estatico, dia_semana, temp, lluvia, evento, dic_zonas):
    data_grid = [Row(
        pulocationid=int(z), day_of_week=int(dia_semana), pickup_hour=int(h),
        temperature_2m=float(temp), precipitation=float(lluvia), snowfall=0.0, hay_evento=int(evento)
    ) for z in range(1, 265) for h in range(24)]
    
    df_grid = spark.createDataFrame(data_grid)
    df_input = df_grid.join(df_estatico, on="pulocationid", how="left").fillna(0)
    
    predicciones = modelo.transform(df_input)
    df_pd = predicciones.select("pulocationid", "pickup_hour", "prediction").toPandas()
    df_pd['nombre_zona'] = df_pd['pulocationid'].map(lambda x: dic_zonas.get(x, f"Zona {x}"))
    df_pd['pulocationid_str'] = df_pd['pulocationid'].astype(str)
    
    df_pd['prediction'] = df_pd['prediction'].apply(lambda x: max(0, x)).round(0)
    
    return df_pd.sort_values('pickup_hour')

# --- INTERFAZ WEB ---
def main():
    st.title("🗺️ Mapa Interactivo: NYC Taxi Digital Twin")
    st.markdown("Observa cómo evoluciona geográficamente la demanda de taxis al cambiar el clima y los eventos. Dale al botón **Play** debajo del mapa para ver la evolución horaria.")

    with st.spinner("Arrancando motores y verificando modelos AI (Paciencia la primera vez)..."):
        spark, modelo, df_estatico = iniciar_spark_y_modelo()
        dic_zonas = obtener_diccionario_zonas()
        mapa_geojson = obtener_geometria_mapa()

    st.sidebar.header("🎛️ Controles del Mapa")
    
    dias_map = {"Lunes": 2, "Martes": 3, "Miércoles": 4, "Jueves": 5, "Viernes": 6, "Sábado": 7, "Domingo": 1}
    dia_seleccionado = st.sidebar.selectbox("📅 Día de la semana", list(dias_map.keys()))
    
    st.sidebar.markdown("---")
    temp = st.sidebar.slider("🌡️ Temperatura (ºC)", min_value=-15.0, max_value=40.0, value=15.0, step=0.5)
    lluvia = st.sidebar.slider("🌧️ Lluvia (mm/h)", min_value=0.0, max_value=30.0, value=0.0, step=1.0)
    evento = st.sidebar.toggle("🎟️ Activar Gran Evento")
    
    df_mapa = simular_demanda_mapa(spark, modelo, df_estatico, dias_map[dia_seleccionado], temp, lluvia, evento, dic_zonas)
    
    max_viajes = df_mapa['prediction'].max()

    escala_personalizada = [
        [0.0, "rgb(40, 40, 40)"], 
        [0.05, "rgb(120, 120, 120)"],
        [0.1, "#4b0c6b"], 
        [0.3, "#781c6d"], 
        [0.5, "#a52c60"], 
        [0.7, "#ed6925"], 
        [1.0, "#fcffa4"] 
    ]

    fig = px.choropleth_mapbox(
        df_mapa, geojson=mapa_geojson, locations='pulocationid_str', featureidkey='properties.locationid', 
        color='prediction', animation_frame='pickup_hour', hover_name='nombre_zona', 
        hover_data={'pulocationid_str': False, 'pickup_hour': False, 'prediction': True},
        color_continuous_scale=escala_personalizada, range_color=[0, max_viajes],
        mapbox_style="carto-darkmatter", zoom=9.5, center={"lat": 40.7128, "lon": -73.93} 
    )

    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        height=700,
        coloraxis_colorbar=dict(
            title="Viajes", thicknessmode="pixels", thickness=15, 
            yanchor="top", y=0.9, ticks="outside"
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()