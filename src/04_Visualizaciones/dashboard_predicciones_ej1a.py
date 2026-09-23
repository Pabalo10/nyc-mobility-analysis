import os
import sys
import json
import urllib.request
import platform # <--- NUEVO: Para detectar el Sistema Operativo
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from pyspark.sql import SparkSession, Row
from pyspark.ml import PipelineModel

def create_spark_session():
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    
    # --- PARCHE MULTIPLATAFORMA ---
    # Solo aplicamos la ruta C:/ si detectamos que estamos en Windows
    if platform.system() == "Windows":
        os.environ['HADOOP_HOME'] = "C:/hadoop"
        
    spark = SparkSession.builder.appName("Visualizacion_Mapa_WebApp_Pro").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def cargar_modelo(spark):
    ruta_modelo = str(Path(__file__).resolve().parents[1] / "modelos" / "mejor_modelo_demanda")
    print(f"Cargando modelo entrenado desde: {ruta_modelo}")
    return PipelineModel.load(ruta_modelo)

def obtener_nombres_zonas():
    url_oficial = "https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv"
    try:
        df_zonas = pd.read_csv(url_oficial)
        return dict(zip(df_zonas['LocationID'], df_zonas['Zone']))
    except Exception: return {}

def obtener_geometria_mapa():
    print("Descargando polígonos de NYC...")
    url_geojson = "https://data.cityofnewyork.us/api/geospatial/8meu-9t5y?method=export&format=GeoJSON"
    req = urllib.request.Request(url_geojson, headers={'User-Agent': 'Mozilla/5.0'})
    return json.loads(urllib.request.urlopen(req).read())

# AJUSTE PARA EL NUEVO MODELO: Añadido dataset_completo y clima simulado
def generar_datos_dia(spark, model, dataset_completo, dia_semana, diccionario_zonas):
    # 1. Extraemos los restaurantes y pisos del dataset base
    columnas_estaticas = ["pulocationid", "num_restaurantes", "precio_medio_rest", "num_alquileres", "precio_medio_alquiler"]
    datos_estaticos_zonas = dataset_completo.select(columnas_estaticas).dropDuplicates(["pulocationid"])

    # 2. Creamos la "rejilla" con el clima por defecto
    data_grid = [Row(
        pulocationid=int(z), 
        day_of_week=int(dia_semana), 
        pickup_hour=int(h),
        temperature_2m=15.0, precipitation=0.0, snowfall=0.0, hay_evento=0
    ) for z in range(1, 265) for h in range(24)]
    
    df_grid = spark.createDataFrame(data_grid)
    
    # 3. Unimos ambas partes y predecimos
    df_input = df_grid.join(datos_estaticos_zonas, on="pulocationid", how="left").fillna(0)
    predicciones = model.transform(df_input)
    
    df_pandas = predicciones.select("pulocationid", "day_of_week", "pickup_hour", "prediction").toPandas()
    df_pandas['prediction'] = df_pandas['prediction'].round(0)
    df_pandas['nombre_zona'] = df_pandas['pulocationid'].map(lambda x: diccionario_zonas.get(x, f"Zona {x}"))
    df_pandas['pulocationid_str'] = df_pandas['pulocationid'].astype(str)
    return df_pandas.sort_values('pickup_hour')

def construir_web_app(spark, modelo, dict_zonas, mapa_geojson, dataset_completo):
    # SE MANTIENE TU RUTA DE SALIDA INTACTA
    base_dir = Path(__file__).resolve().parent
    carpeta_salida = base_dir.parent / "visualizacion" / "Prediccion_Demanda_E1a"
    carpeta_mapas = carpeta_salida / "mapas_diarios"
    
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    carpeta_mapas.mkdir(parents=True, exist_ok=True)
    
    dias = {1: "Domingo", 2: "Lunes", 3: "Martes", 4: "Miércoles", 5: "Jueves", 6: "Viernes", 7: "Sábado"}
    
    print("Analizando datos de toda la semana para calibrar la leyenda global...")
    all_days_data = []
    for d_id in dias.keys():
        all_days_data.append(generar_datos_dia(spark, modelo, dataset_completo, d_id, dict_zonas))
    
    full_df = pd.concat(all_days_data)
    max_global = full_df['prediction'].max()
    print(f"Leyenda calibrada al pico máximo semanal: {max_global} viajes.")

    for dia_id, nombre_dia in dias.items():
        print(f"Generando y sobreescribiendo mapa pro para: {nombre_dia}")
        df_dia = full_df[full_df['day_of_week'] == dia_id]
        
        fig = px.choropleth_mapbox(
            df_dia, geojson=mapa_geojson, locations='pulocationid_str', featureidkey='properties.locationid', 
            color='prediction', animation_frame='pickup_hour', hover_name='nombre_zona', 
            hover_data={'pulocationid_str': False, 'pickup_hour': False, 'prediction': True},
            color_continuous_scale="Inferno", range_color=[0, max_global],
            mapbox_style="carto-darkmatter", zoom=9.8, center={"lat": 40.7128, "lon": -73.93} 
        )

        fig.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            coloraxis_colorbar=dict(title="Viajes", thicknessmode="pixels", thickness=15, lenmode="fraction", len=0.6, yanchor="top", y=0.9, ticks="outside"),
            sliders=[{"currentvalue": {"prefix": "HORA: ", "font": {"color": "black", "size": 18}, "offset": 15}, "pad": {"b": 20, "t": 0}, "len": 0.8, "x": 0.1, "y": 0, "bgcolor": "rgba(255,255,255,0.6)", "activebgcolor": "black"}]
        )
        
        ruta_mapa_dia = carpeta_mapas / f"mapa_{nombre_dia}.html"
        fig.write_html(str(ruta_mapa_dia), config={'scrollZoom': True})

    print("\nGenerando Dashboard Maestro...")
    html_maestro = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>NYC Taxi Predictor Pro</title>
        <style>
            body {{ margin: 0; padding: 0; font-family: 'Segoe UI', sans-serif; background-color: #000; color: white; overflow: hidden; }}
            .header {{ background-color: #111; padding: 10px 25px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; z-index: 100; position: relative; }}
            h1 {{ margin: 0; font-size: 20px; color: #f39c12; text-transform: uppercase; letter-spacing: 2px; }}
            .controles {{ display: flex; align-items: center; gap: 15px; }}
            select {{ padding: 8px 12px; font-size: 14px; border-radius: 4px; background-color: #222; color: white; border: 1px solid #444; outline: none; cursor: pointer; }}
            select:hover {{ border-color: #f39c12; }}
            iframe {{ width: 100vw; height: calc(100vh - 55px); border: none; display: block; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚕 NYC Taxi Demand Dashboard</h1>
            <div class="controles">
                <select id="daySelector" onchange="cambiarMapa()">
                    <option value="mapas_diarios/mapa_Lunes.html" selected>Lunes</option>
                    <option value="mapas_diarios/mapa_Martes.html">Martes</option>
                    <option value="mapas_diarios/mapa_Miércoles.html">Miércoles</option>
                    <option value="mapas_diarios/mapa_Jueves.html">Jueves</option>
                    <option value="mapas_diarios/mapa_Viernes.html">Viernes</option>
                    <option value="mapas_diarios/mapa_Sábado.html">Sábado</option>
                    <option value="mapas_diarios/mapa_Domingo.html">Domingo</option>
                </select>
            </div>
        </div>
        <iframe id="mapFrame" src="mapas_diarios/mapa_Lunes.html"></iframe>
        <script>
            function cambiarMapa() {{
                const selector = document.getElementById('daySelector');
                document.getElementById('mapFrame').src = selector.value;
            }}
        </script>
    </body>
    </html>
    """
    
    ruta_maestra = carpeta_salida / "DASHBOARD_GEOGRAFICO_NYC.html"
    with open(ruta_maestra, "w", encoding="utf-8") as f:
        f.write(html_maestro)
        
    print(f"\n¡Proceso finalizado con éxito!")
    print(f"La estructura de archivos está lista. Abre este archivo en tu navegador:")
    print(f"--> {ruta_maestra} <--")

if __name__ == "__main__":
    spark = create_spark_session()
    try:
        modelo = cargar_modelo(spark)
        dict_zonas = obtener_nombres_zonas()
        mapa_geojson = obtener_geometria_mapa()
        
        # Leemos el parquet desde la raíz para sacar los datos de los restaurantes
        ruta_parquet = str(Path(__file__).resolve().parents[2] / "datos" / "limpios" / "resumen_zona_hora.parquet")
        dataset_completo = spark.read.parquet(ruta_parquet)

        construir_web_app(spark, modelo, dict_zonas, mapa_geojson, dataset_completo)
    except Exception as e: print(f"Error crítico: {e}")
    finally: spark.stop()