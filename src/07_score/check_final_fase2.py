import os
import sys
from pathlib import Path
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

def create_spark_session():
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    os.environ['HADOOP_HOME'] = "C:/hadoop"
    spark = SparkSession.builder.appName("Auditoria_Final_Fase2").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def auditar_dataset_final():
    spark = create_spark_session()
    ruta_dataset = Path(__file__).resolve().parents[2] / "datos" / "limpios" / "dataset_entrenamiento_final_fase2.parquet"

    print("="*60)
    print(" AUDITORÍA PRE-ENTRENAMIENTO - FASE 2")
    print("="*60)

    if not ruta_dataset.exists():
        print(f" ERROR: No se encuentra el archivo en {ruta_dataset}")
        return

    df = spark.read.parquet(str(ruta_dataset))

    # 1. VOLUMEN DE DATOS
    total_filas = df.count()
    print(f" Total de registros listos para entrenar: {total_filas} filas")

    # 2. SANIDAD DEL TARGET (Rentabilidad)
    stats_renta = df.select(
        F.round(F.min("rentabilidad_score"), 2).alias("min"),
        F.round(F.max("rentabilidad_score"), 2).alias("max"),
        F.round(F.avg("rentabilidad_score"), 2).alias("avg")
    ).collect()[0]

    print("\n VARIABLE OBJETIVO (Rentabilidad $/min):")
    print(f"   Min: {stats_renta['min']} | Max: {stats_renta['max']} | Media: {stats_renta['avg']}")
    if stats_renta['max'] >= 15.0:
        print("    ALERTA ROJA: El filtro ha fallado. Hay valores >= 15$/min.")
    else:
        print("    Filtro de outliers OK (Max < 15$/min).")

    # 3. COMPROBACIÓN DE NULOS (Vital para el Random Forest)
    print("\n🕳️ BÚSQUEDA DE VALORES NULOS (NULLS):")
    nulos = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]).collect()[0].asDict()
    hay_nulos = False
    for col, count in nulos.items():
        if count > 0:
            print(f"    Columna '{col}' tiene {count} nulos.")
            hay_nulos = True
    if not hay_nulos:
        print("    Dataset impecable. 0 valores nulos en todas las columnas.")

    # 4. DISTRIBUCIÓN DE RENTABILIDAD (Para ver qué va a aprender el modelo)
    print("\n DISTRIBUCIÓN DE LA RENTABILIDAD (Clases lógicas):")
    df.withColumn("rango",
        F.when(F.col("rentabilidad_score") < 1.0, "1. < 1.0 $/min (Baja)")
         .when((F.col("rentabilidad_score") >= 1.0) & (F.col("rentabilidad_score") < 3.0), "2. 1.0 - 3.0 $/min (Normal)")
         .otherwise("3. > 3.0 $/min (Alta)")
    ).groupBy("rango").count().orderBy("rango").show(truncate=False)

    print("="*60)
    spark.stop()

if __name__ == "__main__":
    auditar_dataset_final()