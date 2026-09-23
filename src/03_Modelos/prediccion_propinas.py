'''
    ============================================================================
    MODELO PREDICTOR DE PROPINA A PRIORI
    ============================================================================

    Este script carga los datos y entrena un modelo de Aprendizaje Automático
    para predecir la propina que se dará al conductor en un determinado viaje,
    antes de que se realice dicho viaje (funcionará tanto para predecir propinas
    para LTC y FHV). 

'''
# ==============================================================================
# PASO 1: IMPORTAR LIBRERÍAS Y ENTORNO (COPIADO DEL SCRIPT QUE FUNCIONA)
# ==============================================================================

import os
import sys
import warnings
from typing import Tuple
from pathlib import Path

import pandas as pd
import geopandas as gpd
import folium
from folium import plugins
import branca.colormap as cm
from scipy import stats
from dotenv import load_dotenv, find_dotenv # Añadido de tu script

# --- CONFIGURACIÓN DE ENTORNO IDÉNTICA A LA TUYA ---
load_dotenv(find_dotenv())
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
#os.environ['HADOOP_HOME'] = "C:/hadoop"
#os.environ['HADOOP_TMP_DIR'] = "C:/tmp/hadoop"
# ¡HEMOS ELIMINADO LA DECLARACIÓN MANUAL DE SPARK_HOME!
# ---------------------------------------------------

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

warnings.filterwarnings('ignore')

# ==============================================================================
# PASO 2: CONFIGURACIÓN DE CREDENCIALES MINIO / S3
# ==============================================================================

# Credenciales MinIO (evitar hardcodear secretos)
MINIO_CONFIG = {
    "endpoint": os.getenv("MINIO_ENDPOINT", "https://minio.fdi.ucm.es"),
    "access_key": os.getenv("MINIO_ACCESS_KEY", ""),
    "secret_key": os.getenv("MINIO_SECRET_KEY", ""),
    "path_style": os.getenv("MINIO_PATH_STYLE", "true")
}

# Rutas de datos en el bucket
S3_PATHS = {
    "taxi": os.getenv("MINIO_TAXI_PATH", "s3a://pd2/taxomanos/limpios/nyc_taxi_clean.parquet"),
    "fhv": os.getenv("MINIO_FHV_PATH", "s3a://pd2/taxomanos/limpios/fhv_2023_clean.parquet")
}

# Rutas Locales (Backup)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOCAL_PATHS = {
    "taxi": os.getenv("LOCAL_TAXI_PATH", os.path.join(PROJECT_ROOT, "datos", "limpios", "nyc_taxi_clean.parquet")),
    "fhv": os.getenv("LOCAL_FHV_PATH", os.path.join(PROJECT_ROOT, "datos", "limpios", "fhv_2023_clean.parquet"))
}

RESTAURANTS_PATH = os.getenv("RESTAURANTS_CSV_PATH", os.path.join(PROJECT_ROOT, "datos", "crudos", "restaurantes_nyc_clean.csv"))
K_DIAGNOSTIC_PNG = os.path.join(PROJECT_ROOT, "outputs", "kmeans_elbow_silhouette.png")
# URL del shapefile oficial de zonas de taxi de NYC
TAXI_ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"


# ==============================================================================
# PASO 3: CREAR Y CONFIGURAR SESIÓN SPARK PARA MINIO (VERSIÓN CORREGIDA)
# ==============================================================================

def crear_spark_session() -> SparkSession:
    print("=" * 60)
    print("PASO 3: Configurando sesión de Spark para MinIO...")
    print("=" * 60)
    
    packages = "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262"
    
    spark = SparkSession.builder \
        .appName("NYC_Taxi_Prediccion_Propinas") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.jars.packages", packages) \
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_CONFIG["endpoint"]) \
        .config("spark.hadoop.fs.s3a.access.key", MINIO_CONFIG["access_key"]) \
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_CONFIG["secret_key"]) \
        .config("spark.hadoop.fs.s3a.path.style.access", MINIO_CONFIG["path_style"]) \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem") \
        .config("spark.local.dir", os.environ.get('HADOOP_TMP_DIR', "C:/tmp/hadoop")) \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    print(f"✓ Spark Session creada con éxito")
    return spark

import os
import shutil
def limpiar_tmp_spark(tmp_dir="C:/tmp/hadoop"):
    """
    Limpia el contenido temporal de Spark sin borrar la carpeta base
    """

    if not os.path.exists(tmp_dir):
        return

    print(f"🧹 Limpiando contenido de: {tmp_dir}")

    for item in os.listdir(tmp_dir):
        item_path = os.path.join(tmp_dir, item)

        try:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path, ignore_errors=True)
            else:
                os.remove(item_path)
        except Exception as e:
            print(f"⚠️ No se pudo borrar {item_path}: {e}")

    print("✅ Limpieza completada")

# ==============================================================================
# PASO 4: CARGAR DATOS (INTENTO MINIO -> BACKUP LOCAL)
# ==============================================================================

def cargar_dataset(spark: SparkSession, nombre_clave: str):
    """
    Intenta cargar datos desde MinIO. Si falla, carga desde el backup local.
    """
    print(f"\n--- Cargando dataset: {nombre_clave} ---")
    
    # 1. Intento con MinIO
    #try:
    #    print(f"Intentando cargar desde MinIO: {S3_PATHS[nombre_clave]}...")
    #    df = spark.read.parquet(S3_PATHS[nombre_clave])
    #    print(f"✓ {nombre_clave.capitalize()} cargado exitosamente desde MinIO.")
    #    return df
    #except Exception as e:
    #    print(f"⚠ Fallo en MinIO: {str(e).splitlines()[0]}") # Solo mostramos la primera línea del error
        
    # 2. Intento con Local (Backup)
    try:
        print(f"Recurriendo al backup local: {LOCAL_PATHS[nombre_clave]}...")
        df = spark.read.parquet(LOCAL_PATHS[nombre_clave])
        print(f"✓ {nombre_clave.capitalize()} cargado desde disco local.")
        return df
    except Exception as e:
        print(f"❌ Error crítico: No se encontró el dataset en MinIO ni en Local.")
        raise e

def cargar_datos_taxi(spark: SparkSession):
    df_taxi = cargar_dataset(spark, "taxi")
    
    # --- NORMALIZACIÓN DE COLUMNAS ---
    if "pulocationid" in df_taxi.columns:
        df_taxi = df_taxi.withColumnRenamed("pulocationid", "PULocationID")
    
    columnas_taxi = ["PULocationID", #numérica a la vista pero categórica en ralidad: NO tratar como numérica
                    "tpep_pickup_datetime", 
                    "tip_amount",
                    "passenger_count",
                    "fare_amount",
                    "total_amount"
                    ]

    columnas_existentes = [col for col in columnas_taxi if col in df_taxi.columns]
    
    # --- BUG CORREGIDO (Era df_taxi, no df) ---
    df_taxi = df_taxi.filter(F.col("tip_amount") >= 0)
    df_taxi = df_taxi.filter(F.col("fare_amount") > 0)
    df_taxi = df_taxi.filter(F.col("total_amount") > 0)

    df_taxi = df_taxi.select(columnas_existentes)
    df_taxi = df_taxi.withColumn("tipo_servicio", F.lit("taxi"))
    
    print(f"✓ Datos de taxi normalizados y cargados.")
    return df_taxi

def cargar_datos_fhv(spark: SparkSession):
    df_fhv = cargar_dataset(spark, "fhv")
    
    if "PULocationID" in df_fhv.columns:
        df_fhv = df_fhv.select("PULocationID")
    elif "pickup_location_id" in df_fhv.columns:
        df_fhv = df_fhv.select(F.col("pickup_location_id").alias("PULocationID"))
    
    df_fhv = (df_fhv
        .withColumn("tip_amount", F.lit(0.0).cast(DoubleType()))
        .withColumn("passenger_count", F.lit(1.0).cast(DoubleType()))
        .withColumn("total_amount", F.lit(0.0).cast(DoubleType()))
        .withColumn("tipo_servicio", F.lit("fhv"))
    )
    return df_fhv


# ==============================================================================
# PASO 5: FEATURE ENGINEERING
# ==============================================================================

def creacion_variables(df_taxi):
    print("\n" + "=" * 60 )
    print("PASO 5: Creando variables auxiliares para el modelo ...")
    print("=" * 60 )

    # Creación de variables auxiliares temporales
    df_auxt = df_taxi.withColumn(
        "hour", F.hour("tpep_pickup_datetime")
    ).withColumn(
        "day_of_week", F.dayofweek("tpep_pickup_datetime")
    ).withColumn(
        "is_weekend",
        F.when(F.col("day_of_week").isin([1, 7]), True).
        otherwise(False)
    ).withColumn(
        "month", F.month("tpep_pickup_datetime")
    )

    # Creación de variables auxiliares cuantitativas
    df_final = df_auxt.withColumn(
        "tip_pct", (F.col("tip_amount") / F.col("fare_amount")) * 100
    )

    df_final = df_final.withColumn(
        "tip_class",
        F.when(F.col("tip_pct") == 0, 0)
        .when(F.col("tip_pct") < 10, 1)
        .when(F.col("tip_pct") < 20, 2)
        .otherwise(3)
    )

    return df_final

def crear_features_avanzadas(train_df, test_df):
    """
    Crea features avanzadas A PRIORI basadas en históricos (sin leakage - usando solo train)
    """

    print("🚀 Creando features avanzadas...")

    train_df = train_df.withColumn("zone_group", (F.col("PULocationID")/10).cast("int"))
    test_df  = test_df.withColumn("zone_group", (F.col("PULocationID")/10).cast("int"))

    avg_tip_zone = train_df.groupBy("zone_group").agg(
        F.mean("tip_pct").alias("avg_tip_zone")
    )

    train_df = train_df.join(avg_tip_zone, on="zone_group", how="left")
    test_df  = test_df.join(avg_tip_zone, on="zone_group", how="left")

    avg_tip_zone_hour = train_df.groupBy("zone_group", "hour").agg(
        F.mean("tip_pct").alias("avg_tip_zone_hour")
    )

    train_df = train_df.join(avg_tip_zone_hour, ["zone_group","hour"], "left")
    test_df  = test_df.join(avg_tip_zone_hour, ["zone_group","hour"], "left")

    trip_count_zone = train_df.groupBy("zone_group").agg(
        F.count("*").alias("trip_count_zone")
    )

    train_df = train_df.join(trip_count_zone, on="zone_group", how="left")
    test_df  = test_df.join(trip_count_zone, on="zone_group", how="left")

    train_df = train_df.fillna(0)
    test_df  = test_df.fillna(0)

    print("✅ Features avanzadas creadas")

    return train_df, test_df

def estudio_analítico(df_final):
    print("\n" + "=" * 60 )
    print(" Estudio analítico del df final ...")
    print("\n" + "=" * 60 )

    # --- 1) Comprobamos las métricas de cada columna
    numeric_cols = [
                    "tip_amount", 
                    "passenger_count", 
                    "fare_amount",  
                    "total_amount", 
                    "tip_pct"
                    ]

    print("\n 1) Estadísticas descriptivas de las columnas numéricas")
    df_final.select(numeric_cols).describe().show()

    # --- 2) Análisis variable objetivo
    print("\n 2) Análisis variable objetivo ")
    
    # Quitamos los .show() del final y añadimos la F. a min y max
    df_tip_amount = df_final.select(
        F.lit("tip_amount").alias("variable"),
        F.mean("tip_amount").alias("media"),
        F.stddev("tip_amount").alias("desv_tipica"),
        F.min("tip_amount").alias("min"),
        F.max("tip_amount").alias("max")
    )

    df_tip_pct = df_final.select(
        F.lit("tip_pct").alias("variable"),
        F.mean("tip_pct").alias("media"),
        F.stddev("tip_pct").alias("desv_tipica"),
        F.min("tip_pct").alias("min"),
        F.max("tip_pct").alias("max")
    )

    # Unir verticalmente y AHORA sí mostramos el resultado
    df_tip_stats = df_tip_amount.unionByName(df_tip_pct)
    df_tip_stats.show()

    # --- 3) Estudio de correlaciones entre variables
    print("\n 3) Estudio de correlaciones entre variables numéricas y variable objetivo")
    for c in numeric_cols:
        if c != "tip_amount":
            try:
                corr_value = df_final.stats.corr("tip_amount", c)
                print(f"Correlación entre tip_amount y {c}: {corr_value:.2f} \n")
            except:
                pass
    
    # --- 4) Agregaciones temporales para ver como se distribuyen las propinas
    print("\n 4) Agregaciones temporale para observar distribución temporal de las propinas")
    
    print("\n Por Hora")
    df_final.groupBy("hour").agg(
        F.mean("tip_amount").alias("avg_tip_amount"),
        F.count("*").alias("num_trips")
    ).orderBy("hour").show()

    print("\n Por Día de la Semana")
    df_final.groupBy("day_of_week").agg(
        F.mean("tip_amount").alias("avg_tip_amount"),
        F.count("*").alias("num_trips")
    ).orderBy("day_of_week").show()

    print("\n Por Mes")
    df_final.groupBy("month").agg(
        F.mean("tip_amount").alias("avg_tip_amount"),
        F.count("*").alias("num_trips")
    ).orderBy("month").show()

    print("\n Comparación entre semana vs fin de semana")
    # CUIDADO AQUÍ: Tu compañero agrupa por "is_weekday" pero arriba creó la columna "is_weekend"
    df_final.groupBy("is_weekend").agg(
        F.mean("tip_amount").alias("avg_tip_amount"),
        F.count("*").alias("num_trips")
    ).show()

    # --- 5) Comparación entre tarifa y comportamiento con la propina
    print("\n 5) Comparación entre tarifa vs comportamiento con la propina")

    df_final.groupBy("passenger_count").agg(
        F.count("*").alias("num_trips"),
        F.mean("fare_amount").alias("avg_fare"),
        F.mean("tip_pct").alias("avg_tip_pct")
    ).orderBy("passenger_count").show()

# Estudio y análisis de outliers
import matplotlib.pyplot as plt
import seaborn as sns
def generar_boxplots(df_spark, columnas, output_dir):
    """
    Genera boxplots para cada variable numérica y los guarda en disco
    """
    print("\n📊 Generando boxplots...")

    # Convertimos a pandas (solo columnas necesarias)
    df_pd = df_spark.select(columnas).toPandas()

    os.makedirs(output_dir, exist_ok=True)

    for col in columnas:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df_pd[col])
        plt.title(f"Boxplot - {col}")

        filepath = os.path.join(output_dir, f"boxplot_{col}.png")
        plt.savefig(filepath)
        plt.close()

    print(f"✅ Boxplots guardados en: {output_dir}")


def tratar_outliers(df_spark):
    """
    Genera boxplots y elimina outliers usando método IQR
    """

    print("\n🚀 Tratamiento de outliers iniciado...")

    # Columnas numéricas a analizar
    columnas = [
        "tip_amount",
        "fare_amount",
        "total_amount",
        "tip_pct"
    ]

    # Ruta de salida (ajustada a tu estructura)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = os.path.join(project_root, "src", "Visualizacion", "Boxplots_Propinas")

    # 1) Generar boxplots ANTES de limpiar
    #generar_boxplots(df_spark, columnas, output_dir)

    # 2) Cálculo de límites IQR
    bounds = {}

    for col in columnas:
        q1, q3 = df_spark.approxQuantile(col, [0.25, 0.75], 0.01)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        bounds[col] = (lower, upper)


    # 3) Filtrado de outliers
    df_clean = df_spark

    for col in columnas:
        lower, upper = bounds[col]
        df_clean = df_clean.filter(
            (F.col(col) >= lower) & (F.col(col) <= upper)
        )
    
    # 4) Limpiamos "passenger_count"
    df_clean = df_clean[df_clean["passenger_count"] <= 3]

    print("✅ Outliers eliminados correctamente")

    return df_clean


# ==============================================================================
# PASO 6: BASELINE MODEL
# ==============================================================================

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

def lr_pipeline():

    # Variables -a priori-
    categorical_cols = [
        "PULocationID", 
        "day_of_week"
    ]

    analytical_cols = [
        "passenger_count",
        "hour", 
        "month",
        "is_weekend"
    ]

    indexers = [
        StringIndexer(inputCol = col, outputCol = f"{col}_idx", handleInvalid="keep")
        for col in categorical_cols
    ]
    
    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")
        for c in categorical_cols
    ]
    
    assembler = VectorAssembler(
        inputCols = [f"{col}_ohe" for col in categorical_cols] + analytical_cols,
        outputCol = "features"
    )

    # Modelo Base
    lr = LogisticRegression(
        featuresCol = "features",
        labelCol = "tip_class",
        predictionCol = "prediction",
        family = "multinomial"#,
        #maxIter = 100,
        #regParam = 0.1
    )

    evaluator = MulticlassClassificationEvaluator(
        labelCol="tip_class",
        predictionCol="prediction",
        metricName="f1"
    )

    paramGrid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.01, 0.03, 0.1])
        .addGrid(lr.elasticNetParam, [0.0, 0.2])
        .addGrid(lr.maxIter, [50])
        .build()
    )
    
    tvs = TrainValidationSplit(
        estimator=lr,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        trainRatio=0.7,
        seed=42
    )


    pipeline = Pipeline(
        stages = indexers + encoders + [assembler, tvs]
    )

    return pipeline

# ==============================================================================
# PASO 7: PRODUCTION MODEL
# ==============================================================================
def production_pipeline():

    # Variables -a priori-
    categorical_cols = [
        #"zone_group",
        "PULocationID",  
        "day_of_week"
    ]

    numeric_cols = [
        "passenger_count", 
        "fare_amount", 
        "hour",
        "month",
        "is_weekend",
        "avg_tip_zone",
        "avg_tip_zone_hour",
        "trip_count_zone"
    ]

    # Indexación
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
        for col in categorical_cols
    ]

    encoders = [
        OneHotEncoder(inputCol = f"{c}_idx", outputCol = f"{c}_ohe")
        for c in categorical_cols
    ]

    # Ensamblado
    assembler = VectorAssembler(
        inputCols=[f"{col}_ohe" for col in categorical_cols] + numeric_cols,
        outputCol="features"
    )

    # MODELO FINAL
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="tip_class",
        predictionCol="prediction",
        numTrees=50,
        maxDepth=10,
        minInstancesPerNode=10,
        subsamplingRate=0.8,
        featureSubsetStrategy="sqrt",
        seed=42
    )

    pipeline = Pipeline(
        stages=indexers + encoders + [assembler, rf]
    )

    return pipeline

# ==============================================================================
# PASO 8: FUNCIÓN PIPELINE COMPLETO
# ==============================================================================
def data_split(df_final, train_ratio: float = 0.8):
    """
    Divide los datos en entrenamiento y test
    """

    train_df, test_df = df_final.randomSplit([train_ratio, 1-train_ratio], seed=42)

    return train_df, test_df

def train_model(pipeline: Pipeline, train_df):
    """
    Se entrena el modelo con el conjunto de entrenamiento
    """

    model = pipeline.fit(train_df)

    tvs_model = model.stages[-1]

    if hasattr(tvs_model, "bestModel"):
        best_model = tvs_model.bestModel

        print("Mejores hiperparámetros:")
        if hasattr(best_model, "getRegParam"):
            print(f"regParam: {best_model.getRegParam()}")
        if hasattr(best_model, "getElasticNetParam"):
            print(f"elasticNetParam: {best_model.getElasticNetParam()}")
        if hasattr(best_model, "getMaxIter"):
            print(f"maxIter: {best_model.getMaxIter()}")

    return model

def evaluate_model(model, test_df):
    """
    Evalúa el modelo usando varias métricas
    """

    predictions = model.transform(test_df)

    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="tip_class", 
        predictionCol="prediction", 
        metricName="f1"
    )

    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="tip_class",
        predictionCol="prediction", 
        metricName="accuracy"
    )

    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol="tip_class",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )

    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol="tip_class",
        predictionCol="prediction",
        metricName="weightedRecall"
    )

    f1 = evaluator_f1.evaluate(predictions)
    acc = evaluator_acc.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)

    print("\n Evaluación del modelo:")
    print(f"F1: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision ponderada: {precision:.4f}")
    print(f"Recall ponderado: {recall:.4f}")

    return predictions, f1

def run_model(df, modelo):
    """
    Ejecuta todo el flujo:
    preparación, split, entrenamiento y evaluación
    """
    
    if modelo == "baseline":
        print("Starting Logistic Regression Baseline...")
    else:
        print("Starting Random Forest Classifier...")
    
    # 1. Preparar datos
    df = df.drop("tip_amount", "total_amount")

    # 2. Cachear datos (mejora coste temporal)
    df = df.cache()
    df.count()
    
    # 3. Split
    train_df, test_df = data_split(df)

    # 4. Pipeline
    if modelo == "lr":
        pipeline = lr_pipeline()
    else:
        # Features sin leakage -> no necesarias en el baseline
        train_df, test_df = crear_features_avanzadas(train_df, test_df)
        pipeline = production_pipeline()
    
    # 5. Entrenar
    model = train_model(pipeline, train_df)
    
    # 6. Evaluar
    predictions, f1 = evaluate_model(model, test_df)
    
    if modelo == "lr":
        print("✅ LR model completed")
    else:
        print("✅ RFC model completed")
    
    return model, predictions, f1


# ==============================================================================                
# FUNCIÓN PRINCIPAL
# ==============================================================================

def main():
    """
    Función principal que orquesta todo el pipeline de análisis.
    """
    print("\n" + "=" * 80)
    print("  ANÁLISIS DE PROPINAS A PRIORI - NYC TAXI DATA 2023")
    print("=" * 80)
    
    try:

        import time        
        start = time.time()

        spark = crear_spark_session()
        df_taxi = cargar_datos_taxi(spark)
        df_final = creacion_variables(df_taxi)
        df_final = tratar_outliers(df_final)

        estudio_analítico(df_final)

        # -----------------------------------------------------------------------------
        # ENTRENAMOS LOGISTIC REGRESSOR
        # -----------------------------------------------------------------------------
        lr_start = time.time()

        lr_model, lr_predicts, lr_f1 = run_model(df_final, "lr")

        lr_stop = time.time()
        lr_time = (lr_stop - lr_start)/60
        print(f"Tiempo de entrenamiento del Logistic Regressor: {lr_time:.4f} minutos\n")

        # -----------------------------------------------------------------------------
        # ENTRENAMOS PRODUCTION
        # -----------------------------------------------------------------------------
        prod_start = time.time()

        prod_model, prod_predicts, prod_f1 = run_model(df_final, "RFT")
        
        prod_stop = time.time()
        prod_time = (prod_stop - prod_start)/60
        print(f"Tiempo de entrenamiento del Random Forest Classifier: {prod_time:.4f} minutos \n")
        
        # -----------------------------------------------------------------------------
        # Guardamos el modelo
        # -----------------------------------------------------------------------------
        best_model = None
        if lr_f1 >= prod_f1:
            best_model = lr_model
            print(f"Mejor modelo: Logistic Regressor")
        else:
            best_model = prod_model
            print(f"Mejor modelo: Random Forest Classifier")
        
        # Guardamos el modelo en local 
        ruta_modelo_local = Path(__file__).resolve().parents[1] / "modelos" / "mejor_modelo_propinas"
        print(f"Guardando mejor modelo localmente en: {ruta_modelo_local}")
        
        try:
            # .as_uri() genera automáticamente el esquema "file:///" correcto para Mac, Linux o Windows
            ruta_final = ruta_modelo_local.as_uri()
            best_model.write().overwrite().save(ruta_final)
            print(" ¡LOGRADO! Modelo guardado correctamente.")
        except Exception as e:
            print(f"Error persistente al guardar: {e}")

        # -----------------------------------------------------------------------------
        # Finalizamos y limpiamos sesión de Spark
        # -----------------------------------------------------------------------------
        spark.stop()
        limpiar_tmp_spark()

        stop = time.time()
        time = (stop - start)/60
        print(f"Tiempo de ejecución: {time:.4f} minutos \n")


    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    resultado = main()
