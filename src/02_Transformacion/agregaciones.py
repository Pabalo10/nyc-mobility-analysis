import os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import urllib.request
import tempfile
import zipfile  # Añadido para descomprimir de forma segura en Windows

"""
    Script Maestro de Agregación: Taxis + Clima + Eventos + Datos Espaciales.
    Lógica: Prioridad MinIO con Fallback Local ajustado a rutas específicas.
"""

# =====================================================
# CONFIGURACIÓN DE RUTAS Y ENTORNO
# =====================================================
load_dotenv(find_dotenv())
BASE_DIR = Path(__file__).resolve()
PROJECT_ROOT = BASE_DIR.parents[2]
DATOS_DIR = PROJECT_ROOT / "datos"

# Credenciales MinIO
MINIO_CONF = {
    "endpoint": os.getenv("MINIO_ENDPOINT"),
    "key": os.getenv("MINIO_ACCESS_KEY"),
    "secret": os.getenv("MINIO_SECRET_KEY"),
    "bucket": os.getenv("MINIO_BUCKET"),
    "group_path": os.getenv("MINIO_GROUP_PATH")
}

# =====================================================
# FUNCIÓN DE CARGA ROBUSTA (MINIO -> LOCAL)
# =====================================================
def cargar_dataset(nombre_archivo, carpeta_minio, carpeta_local="limpios"):
    ext = nombre_archivo.split('.')[-1]
    ruta_local = DATOS_DIR / carpeta_local / nombre_archivo
    ruta_s3 = f"s3://{MINIO_CONF['bucket']}/{MINIO_CONF['group_path']}/{carpeta_minio}/{nombre_archivo}"
    
    storage_opts = {
        "key": MINIO_CONF["key"],
        "secret": MINIO_CONF["secret"],
        "client_kwargs": {'endpoint_url': MINIO_CONF["endpoint"]}
    }

    try:
        print(f"☁️ Intentando cargar de MinIO ({carpeta_minio}/): {nombre_archivo}")
        if ext == 'parquet':
            return pd.read_parquet(ruta_s3, storage_options=storage_opts)
        else:
            return pd.read_csv(ruta_s3, storage_options=storage_opts)
    except Exception as e:
        print(f"⚠️ Fallo MinIO ({nombre_archivo}): {str(e).splitlines()[0]}")
        print(f"🏠 Cargando desde local ({carpeta_local}/): {ruta_local}")
        if ext == 'parquet':
            return pd.read_parquet(ruta_local)
        else:
            return pd.read_csv(ruta_local)

# =====================================================
# BLOQUES DE PROCESAMIENTO
# =====================================================

def procesar_taxis():
    print("\n--- 🚕 PROCESANDO TAXIS ---")
    df_fhv = cargar_dataset("fhv_2023_clean.parquet", carpeta_minio="limpios", carpeta_local="limpios")
    df_ylc = cargar_dataset("nyc_taxi_clean.parquet", carpeta_minio="limpios", carpeta_local="limpios")

    if "tpep_pickup_datetime" in df_ylc.columns:
        df_ylc = df_ylc.rename(columns={"tpep_pickup_datetime": "pickup_datetime"})

    df_fhv["pickup_datetime"] = pd.to_datetime(df_fhv["pickup_datetime"], errors="coerce")
    df_ylc["pickup_datetime"] = pd.to_datetime(df_ylc["pickup_datetime"], errors="coerce")

    df_fhv = df_fhv.dropna(subset=["pickup_datetime", "pulocationid"])
    df_ylc = df_ylc.dropna(subset=["pickup_datetime", "pulocationid"])

    df_fhv["tipo_servicio"] = "FHV"
    df_ylc["tipo_servicio"] = "YLC"
    
    df_total = pd.concat([df_fhv[["pickup_datetime", "pulocationid", "tipo_servicio"]], 
                          df_ylc[["pickup_datetime", "pulocationid", "tipo_servicio"]]], ignore_index=True)

    df_total["date_only"] = df_total["pickup_datetime"].dt.date
    df_total["pickup_hour"] = df_total["pickup_datetime"].dt.hour.astype(int)
    df_total["day_of_week"] = ((df_total["pickup_datetime"].dt.dayofweek + 1) % 7) + 1
    df_total["pulocationid"] = df_total["pulocationid"].astype(int)
    
    print("📊 Agrupando demanda por Zona, Fecha Exacta y Hora...")
    agg = df_total.groupby(["pulocationid", "date_only", "day_of_week", "pickup_hour", "tipo_servicio"]).size().reset_index(name="viajes")
    pivot = agg.pivot(index=["pulocationid", "date_only", "day_of_week", "pickup_hour"], columns="tipo_servicio", values="viajes").fillna(0).reset_index()

    if "FHV" not in pivot.columns: pivot["FHV"] = 0
    if "YLC" not in pivot.columns: pivot["YLC"] = 0

    pivot["FHV"] = pivot["FHV"].astype(int)
    pivot["YLC"] = pivot["YLC"].astype(int)
    pivot["demanda_viajes"] = pivot["FHV"] + pivot["YLC"]
    
    return pivot.sort_values(["date_only", "pickup_hour", "pulocationid"]).reset_index(drop=True)

def integrar_exogenas(df_base):
    print("\n--- 🌦️ INTEGRANDO CLIMA Y EVENTOS ---")
    
    # Clima
    try:
        weather = cargar_dataset("nyc_weather_2023_first_half.parquet", carpeta_minio="limpios", carpeta_local="limpios")
        weather = weather.reset_index()
        col_time = next((c for c in ['time', 'date', 'Timestamp', 'index'] if c in weather.columns), weather.columns[0])
        
        weather['time_col'] = pd.to_datetime(weather[col_time], errors='coerce')
        weather['date_only'] = weather['time_col'].dt.date
        weather['pickup_hour'] = weather['time_col'].dt.hour
        
        cols_clima = ['date_only', 'pickup_hour']
        for c in ['temperature_2m', 'precipitation', 'snowfall']:
            if c in weather.columns:
                cols_clima.append(c)
            else:
                weather[c] = 0.0 
                cols_clima.append(c)
                
        weather = weather[cols_clima]
        df_base = pd.merge(df_base, weather, on=['date_only', 'pickup_hour'], how='left')
        
        df_base['temperature_2m'] = df_base['temperature_2m'].fillna(15.0) 
        df_base['precipitation'] = df_base['precipitation'].fillna(0.0)
        df_base['snowfall'] = df_base['snowfall'].fillna(0.0)
    except Exception as e: 
        print(f"⚠️ Salto Clima: {e}")

    # Eventos
    try:
        events = cargar_dataset("NYC_events_2023_first_half.parquet", carpeta_minio="limpios", carpeta_local="limpios")
        events = events.reset_index()
        col_date = next((c for c in ['Date', 'Start Date', 'Event Date', 'start_date', 'date'] if c in events.columns), None)
        if not col_date:
            for c in events.columns:
                if 'date' in str(c).lower():
                    col_date = c
                    break
                    
        events['date_only'] = pd.to_datetime(events[col_date], errors='coerce').dt.date
        event_flags = events[['date_only']].drop_duplicates()
        event_flags['hay_evento'] = 1
        
        df_base = pd.merge(df_base, event_flags, on='date_only', how='left')
        df_base['hay_evento'] = df_base['hay_evento'].fillna(0).astype(int)
    except Exception as e: 
        print(f"⚠️ Salto Eventos: {e}")

    print("\n--- 🗺️ INTEGRANDO DATOS ESPACIALES (GEO) ---")
    try:
        import geopandas as gpd
        
        print("🗺️ Descargando Shapefile oficial de la TLC a carpeta temporal...")
        url_shapefile = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
        
        # ---> SOLUCIÓN BLINDADA PARA WINDOWS Y LINUX <---
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = os.path.join(tmpdirname, "taxi_zones.zip")
            
            # 1. Descargamos el ZIP
            urllib.request.urlretrieve(url_shapefile, zip_path)
            
            # 2. Extraemos TODO el contenido físicamente usando Python puro
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdirname)
            
            # 3. Buscamos el archivo .shp extraído
            shp_files = list(Path(tmpdirname).rglob("*.shp"))
            if not shp_files:
                raise FileNotFoundError("No se encontró ningún archivo .shp en el ZIP de la TLC.")
            
            ruta_shp_real = str(shp_files[0])
            
            # 4. Leemos el shapefile nativo sin protocolos conflictivos
            gdf_zonas = gpd.read_file(ruta_shp_real)[['LocationID', 'geometry']].to_crs(epsg=4326)
            
        gdf_zonas['LocationID'] = gdf_zonas['LocationID'].astype(int)

        # Restaurantes
        print("🍽️ Cruzando Restaurantes...")
        df_rest = cargar_dataset("restaurantes_nyc_clean.csv", carpeta_minio="limpios", carpeta_local="limpios")
        gdf_rest = gpd.GeoDataFrame(df_rest, geometry=gpd.points_from_xy(df_rest.Lon, df_rest.Lat), crs="EPSG:4326")
        
        res = gpd.sjoin(gdf_rest, gdf_zonas, how="inner", predicate="intersects")
        rest_agg = res.groupby('LocationID').agg(num_restaurantes=('Name', 'count'), precio_medio_rest=('Price Category', 'mean')).reset_index()
        
        df_base = pd.merge(df_base, rest_agg, left_on='pulocationid', right_on='LocationID', how='left')
        if 'LocationID' in df_base.columns: df_base = df_base.drop(columns=['LocationID'])

        # Real Estate
        print("🏢 Cruzando Pisos y Alquileres...")
        df_re = cargar_dataset("NY Realstate Pricing.csv", carpeta_minio="crudos", carpeta_local="limpios")
        gdf_re = gpd.GeoDataFrame(df_re, geometry=gpd.points_from_xy(df_re.longitude, df_re.latitude), crs="EPSG:4326")
        
        re_agg = gpd.sjoin(gdf_re, gdf_zonas, how="inner", predicate="intersects")
        re_agg = re_agg.groupby('LocationID').agg(num_alquileres=('id', 'count'), precio_medio_alquiler=('price', 'mean')).reset_index()
        
        df_base = pd.merge(df_base, re_agg, left_on='pulocationid', right_on='LocationID', how='left')
        if 'LocationID' in df_base.columns: df_base = df_base.drop(columns=['LocationID'])
        
        # Rellenar nulos
        df_base['num_restaurantes'] = df_base['num_restaurantes'].fillna(0).astype(int)
        df_base['num_alquileres'] = df_base['num_alquileres'].fillna(0).astype(int)
        df_base['precio_medio_rest'] = df_base['precio_medio_rest'].fillna(df_base['precio_medio_rest'].mean()).round(2)
        df_base['precio_medio_alquiler'] = df_base['precio_medio_alquiler'].fillna(df_base['precio_medio_alquiler'].mean()).round(2)

    except ImportError:
        print("\n⚠️ AVISO: Librería 'geopandas' no detectada.")
    except Exception as e:
        print(f"\n⚠️ Fallo en el cruce espacial: {e}. Se continuará sin estas variables.")

    return df_base

# =====================================================
# MAIN: GUARDADO Y SUBIDA
# =====================================================
def main():
    print("🚀 INICIANDO CREACIÓN DEL DATASET MAESTRO (CON MINIO) 🚀\n")
    df_final = integrar_exogenas(procesar_taxis())
    
    # Guardado Local
    out_local = DATOS_DIR / "limpios" / "resumen_zona_hora.parquet"
    out_local.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(out_local, index=False)
    print(f"\n✅ Local guardado: {out_local}")

    # Subida a MinIO
    if all([MINIO_CONF["endpoint"], MINIO_CONF["key"], MINIO_CONF["secret"], MINIO_CONF["bucket"], MINIO_CONF["group_path"]]):
        ruta_s3_out = f"s3://{MINIO_CONF['bucket']}/{MINIO_CONF['group_path']}/limpios/resumen_zona_hora.parquet"
        try:
            print("☁️ Conectando a MinIO para sobreescribir...")
            df_final.to_parquet(ruta_s3_out, index=False, storage_options={
                "key": MINIO_CONF["key"], "secret": MINIO_CONF["secret"],
                "client_kwargs": {'endpoint_url': MINIO_CONF["endpoint"]}
            })
            print(f"✅ MinIO actualizado: {ruta_s3_out}")
        except Exception as e:
            print(f"❌ Error al subir el resultado final a MinIO: {e}")
    else:
        print("⚠️ Faltan credenciales de MinIO. Solo se guardó en local.")

    print("\n🎉 ¡PROCESO COMPLETADO! Dataset enriquecido listo.")

if __name__ == "__main__":
    main()