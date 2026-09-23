import os
import sys
import boto3
from pathlib import Path
from pyspark.sql import SparkSession, Row
from pyspark.ml import PipelineModel
from dotenv import load_dotenv
import pandas as pd

# Cargar credenciales de MinIO
load_dotenv()

def create_spark_session():
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    os.environ['HADOOP_HOME'] = "C:/hadoop"
    
    # Creamos una carpeta temporal dentro de tu proyecto en lugar de usar la de Windows
    temp_dir = str(Path(__file__).resolve().parents[2] / "temp" / "spark")
    os.makedirs(temp_dir, exist_ok=True)
    
    spark = SparkSession.builder \
        .appName("Exportador_Fase2_MinIO") \
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

def intentar_descargar_minio(ruta_local, bucket, object_name):
    """Descarga de MinIO SOLO si el archivo local no existe o está vacío."""
    if ruta_local.exists() and (ruta_local.is_file() and ruta_local.stat().st_size > 1000 or ruta_local.is_dir()):
        print(f" Usando caché local: {ruta_local.name}")
        return

    print(f" Descargando de MinIO: {ruta_local.name} ...")
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
        print(f" No se pudo descargar de MinIO ({e}). Se intentará usar local.")

def subir_carpeta_minio(ruta_carpeta_local, bucket, prefijo_s3):
    """Sube el resultado de Spark (carpeta Parquet) a MinIO."""
    print(f" Subiendo resultado a MinIO: {prefijo_s3} ...")
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
        print(" Sincronización con MinIO completada.")
    except Exception as e:
        print(f" Error al subir a MinIO: {e}")


def resolver_ruta_modelo() -> Path:
    candidates = [
        Path(__file__).resolve().parents[1] / "modelos" / "mejor_modelo_demanda",
        Path(__file__).resolve().parents[1] / "03_Modelos" / "modelos" / "mejor_modelo_demanda",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "No se encontró el modelo de demanda entrenado. "
        "Rutas esperadas: " + ", ".join(str(p) for p in candidates)
    )

def exportar_predicciones_base():
    # 1. RUTAS
    base_dir = Path(__file__).resolve().parents[2] / "datos" / "limpios"
    ruta_resumen = base_dir / "resumen_zona_hora.parquet"
    ruta_salida = base_dir / "demandas_base_fase2.parquet"
    
    ruta_modelo = str(resolver_ruta_modelo())

    # 2. SINCRONIZAR ENTRADA
    intentar_descargar_minio(ruta_resumen, "pd2", "taxomanos/limpios/resumen_zona_hora.parquet")

    spark = create_spark_session()
    print(" Iniciando exportación de demandas...")

    try:
        # 3. CARGAR MODELO Y DATOS
        modelo = PipelineModel.load(ruta_modelo)
        dataset_completo = spark.read.parquet(str(ruta_resumen))
        
        columnas_estaticas = ["pulocationid", "num_restaurantes", "precio_medio_rest", "num_alquileres", "precio_medio_alquiler"]
        df_estatico = dataset_completo.select(columnas_estaticas).dropDuplicates(["pulocationid"])

        # 4. GENERAR MATRIZ (7 días x 24h x 264 zonas)
        print(" Generando matriz de 44.352 combinaciones...")
        data_grid = [Row(
            pulocationid=int(z), day_of_week=int(d), pickup_hour=int(h),
            temperature_2m=15.0, precipitation=0.0, snowfall=0.0, hay_evento=0
        ) for z in range(1, 265) for d in range(1, 8) for h in range(24)]
        
        df_grid = spark.createDataFrame(data_grid)
        df_input = df_grid.join(df_estatico, on="pulocationid", how="left").fillna(0)

        # 5. PREDECIR Y GUARDAR
        print(" Ejecutando modelo Random Forest...")
        predicciones = modelo.transform(df_input)
        
        df_final = predicciones.select(
            "pulocationid", "day_of_week", "pickup_hour", "prediction"
        ).withColumnRenamed("prediction", "demanda_predicha")

        guardar_parquet_con_fallback(df_final, ruta_salida)
        print(f" Guardado local en: {ruta_salida}")

        # 6. SINCRONIZAR SALIDA
        subir_carpeta_minio(ruta_salida, "pd2", "taxomanos/limpios/demandas_base_fase2.parquet")

    except Exception as e:
        print(f" Error crítico en el proceso: {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    exportar_predicciones_base()