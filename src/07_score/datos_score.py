import os
import shutil
from typing import Dict

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ==============================
# CONFIG
# ==============================

MINIO_CONFIG: Dict[str, str] = {
    "endpoint": os.getenv("MINIO_ENDPOINT", "https://minio.fdi.ucm.es"),
    "access_key": os.getenv("MINIO_ACCESS_KEY", ""),
    "secret_key": os.getenv("MINIO_SECRET_KEY", ""),
    "path_style": "true",
}

S3_PATHS = {
    "taxi": "s3a://pd2/taxomanos/limpios/nyc_taxi_clean.parquet",
    "fhv": "s3a://pd2/taxomanos/limpios/fhv_2023_clean.parquet",
    "traffic": "s3a://pd2/taxomanos/limpios/dataset_trafico_vis_ready.parquet",
}

LOCAL_PATHS = {
    "taxi": "datos/limpios/nyc_taxi_clean.parquet",
    "fhv": "datos/limpios/fhv_2023_clean.parquet",
    "traffic": "datos/limpios/dataset_trafico_vis_ready.parquet",
}

OUTPUT_PATH = "datos/limpios/dataset_model.parquet"

DEMAND_CANDIDATE_PATHS = [
    os.getenv("DEMAND_CSV_PATH", ""),
    "datos/limpios/prediccion_demanda_zona_dia_hora.csv",
    "datos/limpios/demanda_zona_dia_hora.csv",
    "datos/limpios/prediccion_maxima_demanda.csv",
    "datos/limpios/resumen_zona_hora.parquet",
]

# ==============================
# SPARK
# ==============================

def create_spark():
    return (
        SparkSession.builder.appName("FinalDataset")
        .master("local[*]")
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262",
        )
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_CONFIG["endpoint"])
        .config("spark.hadoop.fs.s3a.access.key", MINIO_CONFIG["access_key"])
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_CONFIG["secret_key"])
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.hadoop.io.native.lib.available", "false")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "30000")
        .config("spark.hadoop.fs.s3a.connection.timeout", "30000")
        .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60")
        .getOrCreate()
    )


# ==============================
# LOAD CON FALLBACK
# ==============================

def load_with_fallback(spark, key):
    try:
        print(f"Intentando MinIO: {key}")
        return spark.read.parquet(S3_PATHS[key])
    except Exception:
        print(f"Fallback local: {key}")
        return spark.read.parquet(LOCAL_PATHS[key])


def build_day_of_week_expr(column_name: str):
    day_text = F.lower(F.trim(F.col(column_name).cast("string")))
    day_int = F.col(column_name).cast("int")

    day_from_name = (
        F.when(day_text.isin("monday", "lunes"), F.lit(2))
        .when(day_text.isin("tuesday", "martes"), F.lit(3))
        .when(day_text.isin("wednesday", "miercoles", "miércoles"), F.lit(4))
        .when(day_text.isin("thursday", "jueves"), F.lit(5))
        .when(day_text.isin("friday", "viernes"), F.lit(6))
        .when(day_text.isin("saturday", "sabado", "sábado"), F.lit(7))
        .when(day_text.isin("sunday", "domingo"), F.lit(1))
    )

    day_from_int = (
        F.when(day_int.between(0, 6), ((day_int + F.lit(1)) % F.lit(7)) + F.lit(1))
        .when(day_int.between(1, 7), day_int)
    )

    return F.coalesce(day_from_name, day_from_int, F.lit(1))


def load_optional_demand(spark):
    for source_path in DEMAND_CANDIDATE_PATHS:
        if not source_path:
            continue

        is_remote = source_path.startswith("s3a://")
        if not is_remote and not os.path.exists(source_path):
            continue

        try:
            print(f"Intentando demanda: {source_path}")
            if source_path.lower().endswith(".csv"):
                demand_df = (
                    spark.read.option("header", True)
                    .option("inferSchema", True)
                    .csv(source_path)
                )
            else:
                demand_df = spark.read.parquet(source_path)

            return demand_df, source_path
        except Exception as exc:
            first_line = str(exc).splitlines()[0] if str(exc) else "error desconocido"
            print(f"No se pudo cargar demanda desde {source_path}: {first_line}")

    print("⚠️ No se encontró fuente de demanda. Se continúa sin esa predictora.")
    return None, None


# ==============================
# PREPARACIÓN MOVILIDAD
# ==============================

def prepare_mobility(df, is_fhv=False):

    if is_fhv:
        fare_col = "base_passenger_fare" if "base_passenger_fare" in df.columns else "fare_amount"
        tolls_expr = F.coalesce(F.col("tolls"), F.lit(0.0)) if "tolls" in df.columns else F.lit(0.0)
        df = df.withColumnRenamed("pickup_datetime", "pickup_datetime") \
               .withColumnRenamed("PULocationID", "zone") \
               .withColumn("total_amount", F.coalesce(F.col(fare_col), F.lit(0.0)) + tolls_expr)
    else:
        df = df.withColumnRenamed("tpep_pickup_datetime", "pickup_datetime") \
               .withColumnRenamed("PULocationID", "zone")

        if "total_amount" not in df.columns:
            tolls_expr = F.coalesce(F.col("tolls"), F.lit(0.0)) if "tolls" in df.columns else F.lit(0.0)
            fare_expr = F.coalesce(F.col("fare_amount"), F.lit(0.0)) if "fare_amount" in df.columns else F.lit(0.0)
            df = df.withColumn("total_amount", fare_expr + tolls_expr)

    df = df.withColumn("pickup_datetime", F.to_timestamp("pickup_datetime"))

    df = df.withColumn("hour", F.hour("pickup_datetime")) \
           .withColumn("day_of_week", F.dayofweek("pickup_datetime"))

    df = df.withColumn(
        "income_rate",
        F.col("total_amount") / F.col("trip_duration_min")
    )

    return df.select("zone", "hour", "day_of_week", "income_rate")


# ==============================
# PREPARACIÓN TRAFICO
# ==============================

def prepare_traffic(df):

    has_zone_key = False
    if "zone" in df.columns:
        has_zone_key = True
    elif "LocationID" in df.columns:
        df = df.withColumnRenamed("LocationID", "zone")
        has_zone_key = True
    elif "PULocationID" in df.columns:
        df = df.withColumnRenamed("PULocationID", "zone")
        has_zone_key = True

    if "hour" not in df.columns:
        if "hora_entera" in df.columns:
            df = df.withColumnRenamed("hora_entera", "hour")
        elif "timestamp" in df.columns:
            df = df.withColumn("hour", F.hour(F.to_timestamp("timestamp")))
        else:
            df = df.withColumn("hour", F.lit(0))

    if "day_of_week" in df.columns:
        df = df.withColumn("day_of_week", build_day_of_week_expr("day_of_week"))
    elif "timestamp" in df.columns:
        df = df.withColumn("day_of_week", F.dayofweek(F.to_timestamp("timestamp")))
    elif "dia_semana" in df.columns:
        df = df.withColumn("day_of_week", build_day_of_week_expr("dia_semana"))
    else:
        df = df.withColumn("day_of_week", F.lit(1))

    if "traffic" not in df.columns:
        if "Vol" in df.columns:
            df = df.withColumnRenamed("Vol", "traffic")
        else:
            df = df.withColumn("traffic", F.lit(0.0))

    df = df.withColumn("hour", F.col("hour").cast("int")) \
           .withColumn("day_of_week", F.col("day_of_week").cast("int")) \
           .withColumn("traffic", F.col("traffic").cast("double"))

    if has_zone_key:
        df = df.withColumn("zone", F.col("zone").cast("int"))
        join_keys = ["zone", "hour", "day_of_week"]
    else:
        join_keys = ["hour", "day_of_week"]

    df = df.groupBy(*join_keys).agg(F.avg("traffic").alias("traffic"))

    stats = df.agg(
        F.min("traffic").alias("min_t"),
        F.max("traffic").alias("max_t")
    ).collect()[0]

    min_t = stats["min_t"]
    max_t = stats["max_t"]

    if min_t is None or max_t is None or max_t == min_t:
        df = df.withColumn("traffic_norm", F.lit(0.5))
    else:
        df = df.withColumn(
            "traffic_norm",
            (F.col("traffic") - F.lit(min_t)) / F.lit(max_t - min_t)
        )

    return df.select(*join_keys, "traffic_norm"), join_keys


def prepare_demand(df):

    def rename_first(frame, candidates, target_name):
        for name in candidates:
            if name in frame.columns:
                if name != target_name:
                    frame = frame.withColumnRenamed(name, target_name)
                return frame, True
        return frame, False

    df, has_zone = rename_first(
        df,
        ["zone", "LocationID", "locationid", "PULocationID", "pulocationid"],
        "zone",
    )

    df, has_hour = rename_first(
        df,
        ["hour", "pickup_hour", "hora_entera", "pickupHour"],
        "hour",
    )

    if not has_hour:
        if "timestamp" in df.columns:
            df = df.withColumn("hour", F.hour(F.to_timestamp("timestamp")))
        else:
            df = df.withColumn("hour", F.lit(0))

    if "day_of_week" in df.columns:
        df = df.withColumn("day_of_week", build_day_of_week_expr("day_of_week"))
    elif "dia_semana" in df.columns:
        df = df.withColumn("day_of_week", build_day_of_week_expr("dia_semana"))
    elif "date_only" in df.columns:
        df = df.withColumn("day_of_week", F.dayofweek(F.to_date("date_only")))
    elif "timestamp" in df.columns:
        df = df.withColumn("day_of_week", F.dayofweek(F.to_timestamp("timestamp")))
    elif "pickup_datetime" in df.columns:
        df = df.withColumn("day_of_week", F.dayofweek(F.to_timestamp("pickup_datetime")))

    df, has_demand = rename_first(
        df,
        [
            "demand_score",
            "demanda_viajes",
            "demanda",
            "demand",
            "prediction",
            "predicted_demand",
            "total",
            "viajes",
            "trip_count",
        ],
        "demand_score",
    )

    if not has_demand:
        raise ValueError("La fuente de demanda no contiene columna de demanda reconocible.")

    df = df.withColumn("hour", F.col("hour").cast("int")) \
           .withColumn("demand_score", F.col("demand_score").cast("double"))

    join_keys = ["hour"]
    if has_zone:
        df = df.withColumn("zone", F.col("zone").cast("int"))
        join_keys = ["zone", "hour"]

    if "day_of_week" in df.columns:
        df = df.withColumn("day_of_week", F.col("day_of_week").cast("int"))
        join_keys.append("day_of_week")

    df = df.groupBy(*join_keys).agg(F.avg("demand_score").alias("demand_score"))

    return df.select(*join_keys, "demand_score"), join_keys


# ==============================
# MAIN
# ==============================

def write_output_with_fallback(df, output_path: str):
    def _delete_if_exists(path: str):
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)

    _delete_if_exists(output_path)

    try:
        df.coalesce(1).write.mode("overwrite").parquet(output_path)
        return
    except Exception as exc:
        msg = str(exc)
        is_windows_nativeio_error = (
            "NativeIO$Windows.access0" in msg or "UnsatisfiedLinkError" in msg
        )
        if not is_windows_nativeio_error:
            raise

        print("⚠️ Fallo nativo de Hadoop en Windows al escribir Parquet con Spark. Uso fallback local...")

    _delete_if_exists(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf = df.toPandas()

    try:
        pdf.to_parquet(output_path, index=False)
        print(f"✅ Guardado con fallback en Parquet: {output_path}")
    except Exception as parquet_exc:
        csv_path = output_path.replace(".parquet", ".csv")
        _delete_if_exists(csv_path)
        pdf.to_csv(csv_path, index=False)
        print(
            "⚠️ Fallback Parquet no disponible "
            f"({parquet_exc}). Guardado CSV en: {csv_path}"
        )

def main():

    spark = create_spark()

    print("📦 Cargando datos...")
    taxi = load_with_fallback(spark, "taxi")
    fhv = load_with_fallback(spark, "fhv")
    traffic = load_with_fallback(spark, "traffic")
    demand_raw, demand_source = load_optional_demand(spark)

    print("⚙️ Preparando movilidad...")
    taxi = prepare_mobility(taxi, is_fhv=False)
    fhv = prepare_mobility(fhv, is_fhv=True)

    df = taxi.unionByName(fhv)

    print("📊 Agregando...")
    df_agg = df.groupBy("zone", "hour", "day_of_week") \
        .agg(F.avg("income_rate").alias("income_rate"))

    print("🚦 Preparando tráfico...")
    traffic, traffic_join_keys = prepare_traffic(traffic)
    if "zone" not in traffic_join_keys:
        print("⚠️ Tráfico sin llave de zona taxi; merge por hour/day_of_week.")

    print("🔗 Merge final...")
    df_final = df_agg.join(
        traffic,
        on=traffic_join_keys,
        how="left"
    )

    if demand_raw is not None:
        print("📈 Preparando demanda...")
        try:
            demand, demand_join_keys = prepare_demand(demand_raw)
            print(f"🔮 Integrando demanda desde: {demand_source}")
            if "zone" not in demand_join_keys:
                print(f"⚠️ Demanda sin llave de zona; merge por {demand_join_keys}.")
            df_final = df_final.join(
                demand,
                on=demand_join_keys,
                how="left"
            )
        except Exception as exc:
            print(f"⚠️ No se pudo integrar demanda ({exc}). Se usará demand_score=0.0")
            df_final = df_final.withColumn("demand_score", F.lit(0.0))
    else:
        df_final = df_final.withColumn("demand_score", F.lit(0.0))

    df_final = df_final.fillna({
        "traffic_norm": 0.5,
        "demand_score": 0.0,
    })

    print("💾 Guardando dataset final...")
    write_output_with_fallback(df_final, OUTPUT_PATH)

    print("✅ Dataset listo para modelo")


if __name__ == "__main__":
    main()