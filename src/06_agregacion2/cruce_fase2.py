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
    
    # Evitar PermissionError usando carpeta temporal local
    temp_dir = str(Path(__file__).resolve().parents[2] / "temp" / "spark_cruce")
    os.makedirs(temp_dir, exist_ok=True)
    
    spark = SparkSession.builder \
        .appName("Mega_Cruce_Fase2") \
        .config("spark.local.dir", temp_dir) \
        .config("spark.hadoop.io.native.lib.available", "false") \
        .getOrCreate()
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


def cargar_parquet_con_fallback(spark, ruta_entrada: Path, etiqueta: str):
    try:
        return spark.read.parquet(str(ruta_entrada))
    except Exception as exc:
        msg = str(exc)
        if "NativeIO$Windows.access0" not in msg and "UnsatisfiedLinkError" not in msg:
            raise
        print(f" Fallback: error nativo de Hadoop al leer {etiqueta} con Spark. Intentando con pandas...")

    if ruta_entrada.is_dir():
        archivos_parquet = sorted(str(p) for p in ruta_entrada.glob("*.parquet"))
        if not archivos_parquet:
            raise FileNotFoundError(f"No se encontraron archivos parquet válidos en {ruta_entrada}")
        pdf = pd.read_parquet(archivos_parquet)
    else:
        pdf = pd.read_parquet(str(ruta_entrada))
    df_spark = spark.createDataFrame(pdf)
    print(f" {etiqueta} cargado con fallback pandas->Spark ({len(pdf)} filas).")
    return df_spark

def intentar_descargar_minio(ruta_local, bucket, object_name):
    """Descarga de MinIO SOLO si el archivo local no existe o está vacío."""
    if ruta_local.exists() and (ruta_local.is_dir() or ruta_local.stat().st_size > 1000):
        print(f" Usando caché local: {ruta_local.name}")
        return

    print(f" No detectado en local. Sincronizando desde MinIO: {ruta_local.name} ...")
    try:
        s3 = boto3.client('s3',
            endpoint_url=os.getenv('MINIO_ENDPOINT', 'http://127.0.0.1:9000'),
            aws_access_key_id=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
            aws_secret_access_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin')
        )
        ruta_local.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, object_name, str(ruta_local))
        print(f" {ruta_local.name} descargado con éxito.")
    except Exception as e:
        print(f" Error al conectar con MinIO ({e}). Se intentará proceder con local.")

def subir_carpeta_minio(ruta_carpeta_local, bucket, prefijo_s3):
    """Sube la carpeta Parquet completa a MinIO."""
    print(f" Subiendo resultado final a MinIO: {prefijo_s3} ...")
    try:
        s3 = boto3.client('s3',
            endpoint_url=os.getenv('MINIO_ENDPOINT', 'http://127.0.0.1:9000'),
            aws_access_key_id=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
            aws_secret_access_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin')
        )
        for root, dirs, files in os.walk(ruta_carpeta_local):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, ruta_carpeta_local)
                s3_key = f"{prefijo_s3}/{relative_path}".replace("\\", "/")
                s3.upload_file(local_path, bucket, s3_key)
        print(" Sincronización de salida completada.")
    except Exception as e:
        print(f" Error al subir a MinIO: {e}")

def cruzar_y_limpiar():
    # 1. DEFINICIÓN DE RUTAS
    base_dir = Path(__file__).resolve().parents[2] / "datos" / "limpios"
    ruta_renta = base_dir / "rentabilidad_historica_fase2.parquet"
    ruta_demanda = base_dir / "demandas_base_fase2.parquet"
    ruta_salida = base_dir / "dataset_entrenamiento_final_fase2.parquet"

    # 2. SINCRONIZACIÓN DE ENTRADA
    intentar_descargar_minio(ruta_renta, "pd2", "taxomanos/limpios/rentabilidad_historica_fase2.parquet")
    intentar_descargar_minio(ruta_demanda, "pd2", "taxomanos/limpios/demandas_base_fase2.parquet")

    spark = create_spark_session()
    print(" Iniciando Mega-Cruce y limpieza de outliers...")

    try:
        # 3. CARGA DE DATOS
        df_renta = cargar_parquet_con_fallback(spark, ruta_renta, "rentabilidad_historica_fase2")
        df_demanda = cargar_parquet_con_fallback(spark, ruta_demanda, "demandas_base_fase2")

        # 4. EL CRUCE (JOIN)
        print(" Uniendo rentabilidad real con demanda predicha...")
        df_final = df_renta.join(
            df_demanda, 
            on=["pulocationid", "day_of_week", "pickup_hour"], 
            how="inner"
        )

        # 5. LIMPIEZA DE OUTLIERS
        # Filtramos valores absurdos como los 157$/min que vimos en el check.
        # Un valor de 15$/min ya es extremadamente alto (900$/hora), lo usamos como techo.
        print(" Filtrando ruidos y errores de tarifa (Profit Score > 15.0)...")
        conteo_antes = df_final.count()
        df_final = df_final.filter(F.col("rentabilidad_score") < 15.0)
        conteo_despues = df_final.count()
        print(f" Limpieza completada: {conteo_antes - conteo_despues} filas eliminadas.")

        # 6. GUARDADO LOCAL
        guardar_parquet_con_fallback(df_final, ruta_salida)
        print(f" Dataset guardado localmente en: {ruta_salida}")

        # 7. SINCRONIZACIÓN DE SALIDA
        subir_carpeta_minio(ruta_salida, "pd2", "taxomanos/limpios/dataset_entrenamiento_final_fase2.parquet")

        # 8. VALIDACIÓN FINAL
        print("\n TOP 5 Oportunidades Reales (Rentabilidad < 15$/min):")
        df_final.orderBy(F.desc("rentabilidad_score")).select(
            "pulocationid", "day_of_week", "pickup_hour", "rentabilidad_score", "demanda_predicha"
        ).show(5)

    except Exception as e:
        print(f" Error durante el cruce: {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    cruzar_y_limpiar()