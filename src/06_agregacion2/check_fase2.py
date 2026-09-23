import os
import sys
from pathlib import Path
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

def create_spark_session():
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    os.environ['HADOOP_HOME'] = "C:/hadoop"
    spark = SparkSession.builder.appName("Check_Fase2").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def realizar_check():
    spark = create_spark_session()
    base_dir = Path(__file__).resolve().parents[2] / "datos" / "limpios"
    
    ruta_rentabilidad = base_dir / "rentabilidad_historica_fase2.parquet"
    ruta_demandas = base_dir / "demandas_base_fase2.parquet"

    print("="*60)
    print(" AUDITORÍA DE DATOS - FASE 2")
    print("="*60)

    # 1. Verificación de Existencia
    if not ruta_rentabilidad.exists() or not ruta_demandas.exists():
        print(" ERROR: Falta alguno de los archivos Parquet en datos/limpios.")
        return

    df_renta = spark.read.parquet(str(ruta_rentabilidad))
    df_demanda = spark.read.parquet(str(ruta_demandas))

    # 2. Análisis de Rentabilidad (Target)
    print(f"\n ARCHIVO: rentabilidad_historica_fase2.parquet")
    print(f"   Total de combinaciones (Zona/Día/Hora): {df_renta.count()}")
    stats_renta = df_renta.select(
        F.min("rentabilidad_score").alias("min_$/min"),
        F.max("rentabilidad_score").alias("max_$/min"),
        F.avg("rentabilidad_score").alias("avg_$/min"),
        F.avg("duracion_media").alias("avg_duracion")
    ).collect()[0]
    
    print(f"    Rentabilidad: Min {stats_renta['min_$/min']} | Max {stats_renta['max_$/min']} | Media {stats_renta['avg_$/min']:.2f} $/min")
    print(f"    Duración media de viajes: {stats_renta['avg_duracion']:.2f} min")

    # 3. Análisis de Demandas Predichas (Feature)
    print(f"\n ARCHIVO: demandas_base_fase2.parquet")
    print(f"   Total de combinaciones generadas: {df_demanda.count()}")
    stats_dem = df_demanda.select(
        F.max("demanda_predicha").alias("max_dem"),
        F.avg("demanda_predicha").alias("avg_dem")
    ).collect()[0]
    print(f"    Demanda predicha: Max {stats_dem['max_dem']:.2f} | Media {stats_dem['avg_dem']:.2f} viajes/h")

    # 4. Prueba de Cruce (Simulación del Paso 3)
    print("\n PRUEBA DE UNIÓN (Inner Join):")
    df_cruce_test = df_renta.join(df_demanda, on=["pulocationid", "day_of_week", "pickup_hour"], how="inner")
    conteo_cruce = df_cruce_test.count()
    print(f"   Filas resultantes tras el cruce: {conteo_cruce}")
    
    if conteo_cruce == 0:
        print("    ALERTA: El cruce devuelve 0 filas. Revisa que los LocationID coincidan.")
    else:
        print("    El cruce funciona correctamente.")

    # 5. Top 5 Oportunidades "Ocultas" (Mucha rentabilidad, demanda moderada)
    print("\n💎 TOP 5 OPORTUNIDADES (Alta rentabilidad + viajes históricos > 100):")
    df_renta.filter("viajes_historicos > 100")\
            .orderBy(F.desc("rentabilidad_score"))\
            .select("pulocationid", "day_of_week", "pickup_hour", "rentabilidad_score", "viajes_historicos")\
            .show(5)

    print("="*60)
    spark.stop()

if __name__ == "__main__":
    realizar_check()