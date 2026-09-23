import os
import sys
import platform # NUEVO: Para detectar el Sistema Operativo
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, dayofweek
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Row

def create_spark_session():
    load_dotenv(find_dotenv())
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    
    # --- PARCHE MULTIPLATAFORMA ---
    # Solo inyectamos las rutas "C:/" si detectamos que estamos en Windows.
    if platform.system() == "Windows":
        os.environ['HADOOP_HOME'] = "C:/hadoop"
        os.environ['HADOOP_TMP_DIR'] = "C:/tmp/hadoop"

    spark = SparkSession.builder \
        .appName("Prediccion_Demanda_Taxi_Ex1a_Potenciado") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.hadoop.fs.s3a.endpoint", os.getenv("MINIO_ENDPOINT")) \
        .config("spark.hadoop.fs.s3a.access.key", os.getenv("MINIO_ACCESS_KEY")) \
        .config("spark.hadoop.fs.s3a.secret.key", os.getenv("MINIO_SECRET_KEY")) \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def prepare_data(spark):
    """Carga los datos y asegura las columnas necesarias"""
    load_dotenv(find_dotenv())
    minio_bucket = os.getenv("MINIO_BUCKET")
    minio_groupPath = os.getenv("MINIO_GROUP_PATH")
    
    ruta_parquet = f"s3a://{minio_bucket}/{minio_groupPath}/limpios/resumen_zona_hora.parquet"
    project_root = Path(__file__).resolve().parents[2]
    ruta_local = project_root / "datos" / "limpios" / "resumen_zona_hora.parquet"

    try:
        print(f"Intentando leer datos desde MinIO: {ruta_parquet}")
        df_grouped = spark.read.parquet(ruta_parquet)
        df_grouped.count()
        print("Datos cargados exitosamente desde MinIO.")
    except Exception as e:
        print(f"Fallo de conexión con MinIO: {str(e).splitlines()[0]}")
        print(f"Leyendo localmente desde: {ruta_local}")
        df_grouped = spark.read.parquet(str(ruta_local))

    if "day_of_week" not in df_grouped.columns and "date_only" in df_grouped.columns:
        df_grouped = df_grouped.withColumn("day_of_week", dayofweek("date_only"))

    return df_grouped.fillna(0).dropna()

def get_zone_dict():
    """Descarga el catálogo oficial de zonas de la TLC y crea un diccionario"""
    url_oficial = "https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv"
    try:
        print("Descargando catálogo oficial de zonas...")
        df_zonas = pd.read_csv(url_oficial)
        return dict(zip(df_zonas['LocationID'], df_zonas['Zone']))
    except Exception as e:
        print(f"Aviso: No se pudo descargar el catálogo ({e}). Se usarán IDs genéricos.")
        return {}

def evaluate_model(model, dataset):
    """Calcula RMSE, MAE y R2 para un modelo dado sobre un dataset específico"""
    predictions = model.transform(dataset)
    eval_rmse = RegressionEvaluator(labelCol="demanda_viajes", predictionCol="prediction", metricName="rmse")
    eval_mae = RegressionEvaluator(labelCol="demanda_viajes", predictionCol="prediction", metricName="mae")
    eval_r2 = RegressionEvaluator(labelCol="demanda_viajes", predictionCol="prediction", metricName="r2")
    
    return eval_rmse.evaluate(predictions), eval_mae.evaluate(predictions), eval_r2.evaluate(predictions)

def train_and_compare(train_data, val_data):
    """Entrena modelos ultra-potenciados en Train, los evalúa en Validation y devuelve el ganador"""
    indexer_zone = StringIndexer(inputCol="pulocationid", outputCol="zone_idx", handleInvalid="keep")
    indexer_day = StringIndexer(inputCol="day_of_week", outputCol="day_idx", handleInvalid="keep")
    encoder = OneHotEncoder(inputCols=["zone_idx", "day_idx"], outputCols=["zone_vec", "day_vec"])
    
    cols_features = ["pickup_hour", "zone_vec", "day_vec"]
    columnas_dataset = train_data.columns
    exogenas = ["temperature_2m", "precipitation", "snowfall", "hay_evento", 
                "num_restaurantes", "precio_medio_rest", "num_alquileres", "precio_medio_alquiler"]
    
    for col_name in exogenas:
        if col_name in columnas_dataset:
            cols_features.append(col_name)

    assembler = VectorAssembler(inputCols=cols_features, outputCol="features", handleInvalid="keep")

    # --- MODELOS POTENCIADOS Y SEGUROS (Anti-Overfitting) ---
    
    # 1. Random Forest (Más árboles, nodos con mínimo 5 instancias)
    rf = RandomForestRegressor(featuresCol="features", labelCol="demanda_viajes", 
                               numTrees=150, maxDepth=12, maxBins=128, minInstancesPerNode=5, seed=42)
    pipeline_rf = Pipeline(stages=[indexer_zone, indexer_day, encoder, assembler, rf])
    print("\n> Entrenando Random Forest Potenciado (150 árboles)... [Paciencia, puede tardar]")
    model_rf = pipeline_rf.fit(train_data)
    rmse_rf, mae_rf, r2_rf = evaluate_model(model_rf, val_data)

    # 2. Gradient-Boosted Trees (Más iteraciones, profundidad controlada, aprendizaje lento)
    gbt = GBTRegressor(featuresCol="features", labelCol="demanda_viajes", 
                       maxIter=80, maxDepth=7, maxBins=128, minInstancesPerNode=5, stepSize=0.05, seed=42)
    pipeline_gbt = Pipeline(stages=[indexer_zone, indexer_day, encoder, assembler, gbt])
    print("> Entrenando Gradient-Boosted Trees Potenciado (80 iteraciones)... [Paciencia, puede tardar]")
    model_gbt = pipeline_gbt.fit(train_data)
    rmse_gbt, mae_gbt, r2_gbt = evaluate_model(model_gbt, val_data)

    # Mostrar comparativa 
    print("\n" + "="*50)
    print(f" RESULTADOS EN VALIDATION (Modelos Ultra)")
    print(f" RF  -> RMSE: {rmse_rf:.2f} | MAE: {mae_rf:.2f} | R2: {r2_rf:.4f}")
    print(f" GBT -> RMSE: {rmse_gbt:.2f} | MAE: {mae_gbt:.2f} | R2: {r2_gbt:.4f}")
    print("="*50)

    if rmse_rf < rmse_gbt:
        print("GANADOR: Random Forest Potenciado")
        return model_rf
    else:
        print("GANADOR: Gradient-Boosted Trees Potenciado")
        return model_gbt

def predict_max_demand_zone(spark, model, dataset_completo, target_day, target_hour, diccionario_zonas):
    """Predice y muestra la zona con mayor demanda, traduciendo el ID a nombre real"""
    print(f"\n--- Prediciendo demanda para el Día {target_day} a las {target_hour}:00 ---")
    print("Condiciones simuladas: 15ºC, Sin lluvia, Sin eventos.")
    
    # Extraemos información estática de las zonas (restaurantes y alquileres)
    columnas_estaticas = ["pulocationid"]
    if "num_restaurantes" in dataset_completo.columns:
        columnas_estaticas.extend(["num_restaurantes", "precio_medio_rest", "num_alquileres", "precio_medio_alquiler"])
    
    datos_estaticos_zonas = dataset_completo.select(columnas_estaticas).dropDuplicates(["pulocationid"])

    # Creamos la grilla para la predicción con valores por defecto para clima/eventos
    zonas_ids = range(1, 264)
    data_grid = [Row(
        pulocationid=int(z), 
        day_of_week=int(target_day), 
        pickup_hour=int(target_hour),
        temperature_2m=15.0, 
        precipitation=0.0, 
        snowfall=0.0, 
        hay_evento=0
    ) for z in zonas_ids]
    
    df_grid = spark.createDataFrame(data_grid)
    
    # Unimos con los datos estáticos
    if len(columnas_estaticas) > 1:
        df_pred_input = df_grid.join(datos_estaticos_zonas, on="pulocationid", how="left").fillna(0)
    else:
        df_pred_input = df_grid

    predicciones = model.transform(df_pred_input)
    top_zona = predicciones.orderBy(col("prediction").desc()).first()

    if top_zona:
        zona_id = int(top_zona['pulocationid'])
        viajes = round(top_zona['prediction'], 2)
        nombre_real = diccionario_zonas.get(zona_id, "Zona Desconocida")
        
        print(f"LA ZONA RECOMENDADA ES: {nombre_real} (ID: {zona_id})")
        print(f"Viajes esperados (predicción): {viajes}")

if __name__ == "__main__":
    spark = create_spark_session()
    
    print("Preparando y dividiendo datos...")
    dataset = prepare_data(spark)
    
    # DIVISIÓN: Train (80%), Validation (10%), Test (10%) para igualar fuerzas con el Baseline
    train_df, val_df, test_df = dataset.randomSplit([0.8, 0.1, 0.1], seed=42)

    # 1. Entrenar y elegir el mejor modelo usando Validation
    best_model = train_and_compare(train_df, val_df)

    # 2. Examen Final: Evaluar el ganador en Test
    print("\n" + "*"*50)
    print(" EXAMEN FINAL EN SET DE TEST (Batalla vs Baseline)")
    rmse_test, mae_test, r2_test = evaluate_model(best_model, test_df)
    print(f" Rendimiento real -> RMSE: {rmse_test:.2f} | MAE: {mae_test:.2f} | R2: {r2_test:.4f}")
    print("*"*50)

    diccionario_oficial = get_zone_dict()

    # 3. Predicción práctica
    predict_max_demand_zone(spark, best_model, dataset, target_day=2, target_hour=8, diccionario_zonas=diccionario_oficial)

    print("\n" + "-"*50)
    print("PROCESO DE CÁLCULO FINALIZADO EXITOSAMENTE")
    print("-" * 50)

    # 4. Guardar modelo (MODIFICACIÓN MULTIPLATAFORMA)
    # Mantenemos ruta_modelo_local como un objeto Path, NO como string, para poder usar .as_uri()
    ruta_modelo_local = Path(__file__).resolve().parents[1] / "modelos" / "mejor_modelo_demanda"
    print(f"Guardando mejor modelo localmente en: {ruta_modelo_local}")
    
    try:
        # .as_uri() genera automáticamente el esquema "file:///" correcto para Mac, Linux o Windows
        ruta_final = ruta_modelo_local.as_uri()
        best_model.write().overwrite().save(ruta_final)
        print(" ¡LOGRADO! Modelo guardado correctamente.")
    except Exception as e:
        print(f"Error persistente al guardar: {e}")

    spark.stop()