import os
import sys
import platform
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

def subir_modelo_pyspark_s3a():
    """Usa el conector nativo de PySpark para subir el modelo, esquivando el proxy"""
    load_dotenv(find_dotenv())
    
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    
    # Parche OS-Agnostic
    if platform.system() == "Windows":
        os.environ['HADOOP_HOME'] = "C:/hadoop"
        os.environ['HADOOP_TMP_DIR'] = "C:/tmp/hadoop"

    print("\n1. Arrancando motor PySpark con conector Hadoop S3A...")
    spark = SparkSession.builder \
        .appName("Subida_Nativa_Modelo") \
        .config("spark.driver.memory", "4g") \
        .config("spark.hadoop.fs.s3a.endpoint", os.getenv("MINIO_ENDPOINT")) \
        .config("spark.hadoop.fs.s3a.access.key", os.getenv("MINIO_ACCESS_KEY")) \
        .config("spark.hadoop.fs.s3a.secret.key", os.getenv("MINIO_SECRET_KEY")) \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    try:
        # Rutas locales
        base_dir = Path(__file__).resolve().parents[1]
        ruta_modelo_local = base_dir / "modelos" / "mejor_modelo_demanda"
        
        # Rutas MinIO
        bucket = os.getenv("MINIO_BUCKET")
        group_path = os.getenv("MINIO_GROUP_PATH")
        
        # EL TRUCO ESTÁ AQUÍ: Usamos un nombre nuevo para evitar el "FileAlreadyExists"
        ruta_destino_s3a = f"s3a://{bucket}/{group_path}/models/mejor_modelo_demanda_v2"

        print(f"\n2. Cargando modelo local en memoria (súper rápido)...")
        print(f"   Desde: {ruta_modelo_local}")
        modelo = PipelineModel.load(ruta_modelo_local.as_uri())

        print(f"\n3. Escribiendo modelo directamente en MinIO usando el clúster...")
        print(f"   Destino: {ruta_destino_s3a}")
        print("   (Ignora las advertencias rojas de 'WARNING' si salen. Paciencia...)")
        
        # Guardamos en la nueva ruta
        modelo.write().overwrite().save(ruta_destino_s3a)
        
        print("\n=========================================================")
        print(" ¡SUBIDA COMPLETADA CON ÉXITO!")
        print("=========================================================")
        print(f" Tu modelo está sano, salvo y sin trocear en MinIO.")
        print(f" Tus compañeros pueden cargarlo en sus códigos así:")
        print(f" modelo = PipelineModel.load('{ruta_destino_s3a}')")
        
    except Exception as e:
        print(f"\n[X] Error crítico: {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    subir_modelo_pyspark_s3a()