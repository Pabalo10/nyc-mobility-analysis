import os
import sys
import boto3
from pathlib import Path
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from dotenv import load_dotenv
import pandas as pd

# Cargar variables de entorno
load_dotenv()

def create_spark_session():
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    os.environ['HADOOP_HOME'] = "C:/hadoop"
    spark = (
        SparkSession.builder.appName("Calculo_Rentabilidad_Fase2")
        .config("spark.hadoop.io.native.lib.available", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def guardar_parquet_con_fallback(df_spark, ruta_salida: Path):
    try:
        df_spark.write.mode("overwrite").parquet(str(ruta_salida))
        return
    except Exception as exc:
        msg = str(exc)
        if "NativeIO$Windows.access0" not in msg and "UnsatisfiedLinkError" not in msg:
            raise
        print(" Fallback: error nativo de Hadoop en Windows al escribir parquet con Spark.")

    if ruta_salida.exists():
        if ruta_salida.is_dir():
            import shutil
            shutil.rmtree(ruta_salida)
        else:
            ruta_salida.unlink()

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    pdf = df_spark.toPandas()
    pdf.to_parquet(str(ruta_salida), index=False)
    print(f" Guardado con fallback local en: {ruta_salida}")

def intentar_descargar_minio(ruta_local, bucket, object_name):
    """Descarga de MinIO SOLO si el archivo local no existe o está vacío."""

    # 1. Comprobación inteligente (soporta parquet como carpeta)
    if ruta_local.exists() and (ruta_local.is_dir() or ruta_local.stat().st_size > 1000):
        print(f" ¡Caché local detectada! Usando archivo existente: {ruta_local.name}")
        return

    # 2. Si no existe, procedemos con la descarga
    print(f" Archivo no encontrado en local. Descargando de MinIO: {ruta_local.name} ...")
    try:
        s3 = boto3.client('s3',
            endpoint_url=os.getenv('MINIO_ENDPOINT', 'http://127.0.0.1:9000'),
            aws_access_key_id=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
            aws_secret_access_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin')
        )
        ruta_local.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, object_name, str(ruta_local))
        print(f" {ruta_local.name} descargado de MinIO exitosamente.")
    except Exception as e:
        print(f" Falló la descarga de MinIO ({e}). Verifica tu conexión o si el archivo existe en el bucket.")
        # Si llegamos aquí y no hay archivo local, Spark fallará más adelante, 
        # pero es el comportamiento correcto: no podemos inventarnos los datos.

def subir_carpeta_minio(ruta_carpeta_local, bucket, prefijo_s3):
    """Sube una carpeta entera generada por Spark (Parquet) a MinIO."""
    print(f" Sincronizando con MinIO: subiendo a s3://{bucket}/{prefijo_s3} ...")
    try:
        s3 = boto3.client('s3',
            endpoint_url=os.getenv('MINIO_ENDPOINT', 'http://127.0.0.1:9000'),
            aws_access_key_id=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
            aws_secret_access_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin')
        )
        # Recorremos todos los archivos de las particiones dentro de la carpeta Parquet
        for root, dirs, files in os.walk(ruta_carpeta_local):
            for file in files:
                local_path = os.path.join(root, file)
                # Ruta relativa para que dentro del bucket quede bien ordenado
                relative_path = os.path.relpath(local_path, ruta_carpeta_local)
                s3_key = f"{prefijo_s3}/{relative_path}".replace("\\", "/") # Asegurar barras en MinIO
                
                s3.upload_file(local_path, bucket, s3_key)
        print(" Subida a MinIO completada con éxito.")
    except Exception as e:
        print(f" Error al subir a MinIO: {e}")

def generar_target_rentabilidad():
    spark = create_spark_session()
    print(" Iniciando el cálculo del Profit Score y Duraciones (Paso 2)...")

    # --- 1. RUTAS RELATIVAS Y DESCARGA DE MINIO ---
    base_dir = Path(__file__).resolve().parents[2] / "datos" / "limpios"
    ruta_taxi = base_dir / "nyc_taxi_clean.parquet"
    ruta_fhv = base_dir / "fhv_2023_clean.parquet"

    # Intentar traer de MinIO primero (Asumiendo bucket 'pd2' y prefijo 'taxomanos/limpios/')
    intentar_descargar_minio(ruta_taxi, "pd2", "taxomanos/limpios/nyc_taxi_clean.parquet")
    intentar_descargar_minio(ruta_fhv, "pd2", "taxomanos/limpios/fhv_2023_clean.parquet")

    # --- 2. CARGAR Y UNIFICAR ESQUEMAS (Taxis + Ubers) ---
    print(" Procesando Taxis Amarillos...")
    df_taxi = spark.read.parquet(str(ruta_taxi))
    df_taxi = df_taxi.select(
        F.col("tpep_pickup_datetime").alias("pickup"),
        F.col("tpep_dropoff_datetime").alias("dropoff"),
        F.col("PULocationID").alias("pulocationid"),
        F.col("total_amount")
    )

    print(" Procesando Vehículos FHV (Ubers/Lyfts)...")
    df_fhv = spark.read.parquet(str(ruta_fhv))
    
    # Comprobamos si el FHV tiene dinero (normalmente no, así que ponemos Nulo si falta)
    if "total_amount" in [c.lower() for c in df_fhv.columns]:
        fhv_amount = F.col("total_amount")
    else:
        fhv_amount = F.lit(None).cast("double")

    df_fhv = df_fhv.select(
        F.col("pickup_datetime").alias("pickup"),
        F.col("dropOff_datetime").alias("dropoff"),
        F.col("PUlocationID").alias("pulocationid"),
        fhv_amount.alias("total_amount")
    )

    # Juntamos los dos mundos
    print(" Uniendo datasets...")
    df_unido = df_taxi.unionByName(df_fhv, allowMissingColumns=True)

    # --- 3. CÁLCULOS DE TIEMPO Y DINERO ---
    # Calcular duración en minutos
    df_unido = df_unido.withColumn(
        "duracion_minutos",
        (F.unix_timestamp("dropoff") - F.unix_timestamp("pickup")) / 60.0
    )

    # Filtrar viajes ilógicos (viajes de menos de 1 minuto)
    df_unido = df_unido.filter(F.col("duracion_minutos") > 1)

    # Calcular Profit Score ($/min). Solo donde tengamos dinero (los NULLS se quedan en NULL)
    df_unido = df_unido.withColumn(
        "profit_score",
        F.when(F.col("total_amount").isNotNull() & (F.col("total_amount") > 0), 
               F.col("total_amount") / F.col("duracion_minutos")).otherwise(F.lit(None))
    )

    # Extraer día y hora
    df_unido = df_unido.withColumn("day_of_week", F.dayofweek("pickup")) \
                       .withColumn("pickup_hour", F.hour("pickup"))

    # --- 4. AGRUPACIÓN HISTÓRICA ---
    print(" Calculando rentabilidad y duración media por Zona/Día/Hora...")
    # Al hacer avg() sobre profit_score, Spark automáticamente ignora los NULLs del FHV
    df_rentabilidad = df_unido.groupBy("pulocationid", "day_of_week", "pickup_hour").agg(
        F.round(F.avg("profit_score"), 2).alias("rentabilidad_score"), # TARGET
        F.round(F.avg("duracion_minutos"), 2).alias("duracion_media"),
        F.round(F.avg("total_amount"), 2).alias("ingreso_medio"),
        F.count("*").alias("viajes_historicos")
    ).filter(F.col("rentabilidad_score").isNotNull()) # Quitamos zonas que solo tuvieron Ubers y no sabemos su profit

    # --- 5. GUARDAR Y SUBIR A MINIO ---
    ruta_salida = base_dir / "rentabilidad_historica_fase2.parquet"
    guardar_parquet_con_fallback(df_rentabilidad, ruta_salida)
    print(f" Dataset local guardado en: {ruta_salida}")
    
    # Subimos a la nube
    subir_carpeta_minio(ruta_salida, "pd2", "taxomanos/limpios/rentabilidad_historica_fase2.parquet")
    
    print("\n TOP 5 Zonas/Horas más rentables de NYC ($/min) (Mín. 50 viajes):")
    df_rentabilidad.filter(F.col("viajes_historicos") > 50).orderBy(F.desc("rentabilidad_score")).show(5)
    
    spark.stop()

if __name__ == "__main__":
    generar_target_rentabilidad()