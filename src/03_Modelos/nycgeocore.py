import io
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import branca.colormap as cm
import folium
import geopandas as gpd
import pandas as pd
import requests
from branca.element import Element
from dotenv import load_dotenv
from folium import plugins

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window
from pyspark.sql.window import Window


# -----------------------------------------------------------------------------
# Configuración inicial (carga entorno). Incluye fallback por si falla conexión
# -----------------------------------------------------------------------------
load_dotenv()

# Evita que Spark intente usar el alias "python" de Windows Store.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

MINIO_CONFIG: Dict[str, str] = {
    "endpoint": os.getenv("MINIO_ENDPOINT", "https://minio.fdi.ucm.es"),
    "access_key": os.getenv("MINIO_ACCESS_KEY", ""),
    "secret_key": os.getenv("MINIO_SECRET_KEY", ""),
    "path_style": os.getenv("MINIO_PATH_STYLE", "true"),
}

S3_PATHS = {
    "taxi": "s3a://pd2/taxomanos/limpios/nyc_taxi_clean.parquet",
    "fhv": "s3a://pd2/taxomanos/limpios/fhv_2023_clean.parquet",
    "restaurants": "s3a://pd2/taxomanos/limpios/restaurantes_nyc_clean.csv",
}

LOCAL_FALLBACKS = {
    "taxi": "datos/limpios/nyc_taxi_clean.parquet",
    "fhv": "datos/limpios/fhv_2023_clean.parquet",
    "restaurants": "datos/crudos/restaurantes_nyc_clean.csv",
}

RESTAURANTS_LOCAL_CANDIDATES = [
    "datos/crudos/restaurantes_nyc_clean.csv",
]

RENTALS_LOCAL_CANDIDATES = [
    "datos/crudos/NY Realstate Pricing.csv",
    "datos/crudos/NY_Realstate_Pricing.csv",
]

TAXI_ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

OUTPUT_DATA_DIR = Path("outputs")
OUTPUT_MAP_DIR = Path("src/Visualizacion/Cluster_adquisitivo")
OUTPUT_HTML = OUTPUT_MAP_DIR / "mapa_poder_adquisitivo.html"
OUTPUT_PARQUET = OUTPUT_DATA_DIR / "zonas_aggregated.parquet"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("cluster_zonas")


# -----------------------------------------------------------------------------
# Spark session
# -----------------------------------------------------------------------------
def crear_spark_session() -> SparkSession:
    packages = [
        "org.apache.hadoop:hadoop-aws:3.3.4",
        "com.amazonaws:aws-java-sdk-bundle:1.12.262",
    ]

    spark = (
        SparkSession.builder.appName("NYC_Taxi_FHV_Poder_Adquisitivo")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.pyspark.driver.python", sys.executable)
        .config("spark.pyspark.python", sys.executable)
        .config("spark.executorEnv.PYSPARK_PYTHON", sys.executable)
        .config("spark.jars.packages", ",".join(packages))
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_CONFIG["endpoint"])
        .config("spark.hadoop.fs.s3a.access.key", MINIO_CONFIG["access_key"])
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_CONFIG["secret_key"])
        .config("spark.hadoop.fs.s3a.path.style.access", MINIO_CONFIG["path_style"])
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true")
        # Timeouts en milisegundos para evitar parseos ambiguos en Hadoop/S3A.
        .config("spark.hadoop.fs.s3a.connection.timeout", "60000")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "5000")
        .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60")
        .config("spark.hadoop.fs.s3a.connection.ttl", "600000")
        .config("spark.hadoop.fs.s3a.connection.request.timeout", "60000")
        .config("spark.hadoop.fs.s3a.multipart.purge", "false")
        .config("spark.hadoop.fs.s3a.multipart.purge.age", "86400")
        .config("spark.hadoop.fs.s3a.attempts.maximum", "1")
        .config("spark.hadoop.fs.s3a.retry.limit", "1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info("Spark iniciado: %s", spark.version)
    return spark


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def preferir_local() -> bool:
    # En Windows priorizamos local para evitar bloqueos de lectura S3A en entornos docentes.
    mode = os.getenv("PREFER_LOCAL_DATA", "auto").strip().lower()
    if mode in {"1", "true", "yes", "local"}:
        return True
    if mode in {"0", "false", "no", "s3"}:
        return False
    return os.name == "nt"


def resolve_project_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def first_existing_path(candidates) -> Optional[Path]:
    for c in candidates:
        p = resolve_project_path(c)
        if p.exists():
            return p
    return None


def cargar_parquet_con_fallback(spark: SparkSession, key: str) -> DataFrame:
    s3_path = S3_PATHS[key]
    local_path = LOCAL_FALLBACKS.get(key)
    local_abs = resolve_project_path(local_path) if local_path else None

    if preferir_local():
        if local_abs and local_abs.exists():
            logger.info("Leyendo %s desde local (prioridad Windows): %s", key, local_abs)
            return spark.read.parquet(str(local_abs))
        raise FileNotFoundError(
            f"No existe fallback local para {key}: {local_abs}. "
            "En Windows se evita MinIO por defecto para no bloquear la sesion. "
            "Si quieres forzar S3, define PREFER_LOCAL_DATA=s3"
        )

    try:
        logger.info("Leyendo %s desde MinIO: %s", key, s3_path)
        return spark.read.parquet(s3_path)
    except Exception as s3_err:
        logger.warning("Fallo MinIO para %s: %s", key, str(s3_err).splitlines()[0])

    if local_abs and local_abs.exists():
        logger.info("Intentando fallback local: %s", local_abs)
        return spark.read.parquet(str(local_abs))

    raise RuntimeError(f"No se pudo leer dataset {key} desde S3 ni local")


def find_first_existing_column(df: DataFrame, candidates) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


# -----------------------------------------------------------------------------
# Metricas ETL
# -----------------------------------------------------------------------------
def preparar_taxi(df_taxi_raw: DataFrame) -> DataFrame:
    pickup_col = find_first_existing_column(
        df_taxi_raw,
        ["PULocationID", "pulocationid", "pickup_location_id"],
    )
    if not pickup_col:
        raise ValueError("Taxi no tiene columna de pickup_location_id/PULocationID")

    tip_col = find_first_existing_column(df_taxi_raw, ["tip_amount"])
    passenger_col = find_first_existing_column(df_taxi_raw, ["passenger_count"])

    if not tip_col or not passenger_col:
        raise ValueError("Taxi no tiene tip_amount o passenger_count")

    df = (
        df_taxi_raw.select(
            F.col(pickup_col).alias("LocationID"),
            F.col(tip_col).cast(DoubleType()).alias("tip_amount"),
            F.col(passenger_col).cast(DoubleType()).alias("passenger_count"),
        )
        .filter(F.col("LocationID").isNotNull())
        .filter((F.col("LocationID") > 0) & (F.col("LocationID") <= 265))
    )

    taxi_agg = df.groupBy("LocationID").agg(
        F.avg("tip_amount").alias("tip_amount_avg"),
        F.avg("passenger_count").alias("passenger_count_avg"),
        F.count("*").alias("taxi_trip_count"),
    )
    return taxi_agg


def preparar_fhv(df_fhv_raw: DataFrame) -> DataFrame:
    pickup_col = find_first_existing_column(
        df_fhv_raw,
        ["PULocationID", "pulocationid", "pickup_location_id"],
    )
    if not pickup_col:
        raise ValueError("FHV no tiene columna de pickup_location_id/PULocationID")

    df = (
        df_fhv_raw.select(F.col(pickup_col).alias("LocationID"))
        .filter(F.col("LocationID").isNotNull())
        .filter((F.col("LocationID") > 0) & (F.col("LocationID") <= 265))
    )

    fhv_agg = df.groupBy("LocationID").agg(F.count("*").alias("fhv_trip_count"))
    return fhv_agg


def combinar_metricas(taxi_agg: DataFrame, fhv_agg: DataFrame) -> DataFrame:
    df = taxi_agg.join(fhv_agg, on="LocationID", how="full_outer").fillna(
        {
            "tip_amount_avg": 0.0,
            "passenger_count_avg": 0.0,
            "taxi_trip_count": 0,
            "fhv_trip_count": 0,
        }
    )

    df = df.withColumn(
        "volumen_viajes",
        F.col("taxi_trip_count") + F.col("fhv_trip_count"),
    )
    return df


def calcular_indice_poder_adquisitivo(df: DataFrame) -> DataFrame:
    has_alquiler = "alquiler_mediano" in df.columns
    feature_cols = ["tip_amount_avg", "passenger_count_avg", "volumen_viajes"]
    if has_alquiler:
        feature_cols.append("alquiler_mediano")

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_raw",
    )
    df_vec = assembler.transform(df)

    scaler = MinMaxScaler(inputCol="features_raw", outputCol="features_scaled")
    scaler_model = scaler.fit(df_vec)
    df_scaled = scaler_model.transform(df_vec)

    arr_col = vector_to_array("features_scaled")
    norm_col_map = {
        "tip_amount_avg": "tip_norm",
        "passenger_count_avg": "passenger_norm",
        "volumen_viajes": "volumen_norm",
        "alquiler_mediano": "alquiler_norm",
    }

    for i, raw_col in enumerate(feature_cols):
        df_scaled = df_scaled.withColumn(norm_col_map[raw_col], arr_col[i])

    if has_alquiler:
        weights = {
            "tip_norm": 0.35,
            "passenger_norm": 0.15,
            "volumen_norm": 0.25,
            "alquiler_norm": 0.25,
        }
    else:
        weights = {
            "tip_norm": 0.50,
            "passenger_norm": 0.20,
            "volumen_norm": 0.30,
        }

    score_expr = F.lit(0.0)
    for c, w in weights.items():
        score_expr = score_expr + F.col(c) * F.lit(w)

    df_idx = df_scaled.withColumn("poder_adquisitivo", score_expr)

    df_idx = df_idx.withColumn(
        "poder_adquisitivo_0_100", F.col("poder_adquisitivo") * F.lit(100.0)
    )

    return df_idx.drop("features_raw", "features_scaled")


def aplicar_kmeans(df: DataFrame, k: int = 6) -> Tuple[DataFrame, KMeans]:
    feature_cols = ["tip_norm", "passenger_norm", "volumen_norm"]
    if "alquiler_norm" in df.columns:
        feature_cols.append("alquiler_norm")

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="kmeans_features",
    )
    df_vec = assembler.transform(df)

    kmeans = KMeans(
        k=k,
        seed=42,
        featuresCol="kmeans_features",
        predictionCol="cluster",
        maxIter=100,
    )
    model = kmeans.fit(df_vec)
    pred = model.transform(df_vec).drop("kmeans_features")

    # Reordenamos clusters por poder adquisitivo medio: 0 = mas ricos, k-1 = mas pobres.
    cluster_order = (
        pred.groupBy("cluster")
        .agg(F.avg("poder_adquisitivo_0_100").alias("cluster_mean"))
        .orderBy(F.desc("cluster_mean"))
    )
    w = Window.orderBy(F.desc("cluster_mean"))
    cluster_map = cluster_order.withColumn("cluster_rank", F.row_number().over(w) - F.lit(1)).select(
        F.col("cluster").alias("cluster_raw"),
        F.col("cluster_rank").cast("int").alias("cluster"),
    )

    pred = pred.join(cluster_map, pred.cluster == cluster_map.cluster_raw, "left").drop("cluster_raw", pred.cluster)
    pred = pred.withColumn("cluster_label", F.concat(F.lit("Cluster "), F.col("cluster").cast("string")))
    return pred, model


# -----------------------------------------------------------------------------
# Geo + mapa
# -----------------------------------------------------------------------------
def descargar_y_encontrar_shapefile(base_dir: Path = Path("taxi_zones")) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)

    shp_files = list(base_dir.rglob("*.shp"))
    if shp_files:
        best = sorted(shp_files, key=lambda p: (p.name != "taxi_zones.shp", len(str(p))))[0]
        logger.info("Shapefile encontrado en cache: %s", best)
        return best

    logger.info("Descargando shapefile desde: %s", TAXI_ZONES_URL)
    resp = requests.get(TAXI_ZONES_URL, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(base_dir)

    shp_files = list(base_dir.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError("No se encontro .shp tras extraer taxi_zones.zip")

    best = sorted(shp_files, key=lambda p: (p.name != "taxi_zones.shp", len(str(p))))[0]
    logger.info("Shapefile detectado: %s", best)
    return best


def cargar_zonas_gdf() -> gpd.GeoDataFrame:
    shp_path = descargar_y_encontrar_shapefile()
    gdf = gpd.read_file(shp_path)

    if "LocationID" not in gdf.columns:
        for c in ["OBJECTID", "objectid", "locationid", "LocationId"]:
            if c in gdf.columns:
                gdf = gdf.rename(columns={c: "LocationID"})
                break

    if "LocationID" not in gdf.columns:
        raise ValueError("El shapefile no contiene LocationID/OBJECTID")

    gdf["LocationID"] = gdf["LocationID"].astype(int)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    return gdf


def cargar_restaurantes_filtrados(
    spark: SparkSession,
    gdf_zonas: Optional[gpd.GeoDataFrame] = None,
    max_points: int = 1500,
) -> Optional[gpd.GeoDataFrame]:
    try:
        source = S3_PATHS["restaurants"]
        local_rest_abs = first_existing_path(RESTAURANTS_LOCAL_CANDIDATES)

        if preferir_local():
            if local_rest_abs:
                source = str(local_rest_abs)
            else:
                logger.info("No hay CSV local de restaurantes; se omite capa de restaurantes.")
                return None

        # Si es local, usamos pandas para evitar stages extra de Spark en Windows.
        if str(source).lower().startswith("s3a://"):
            df = spark.read.option("header", True).option("inferSchema", True).csv(source)

            rating_col = find_first_existing_column(df, ["rating", "stars", "score"])
            lat_col = find_first_existing_column(df, ["latitude", "lat", "y"])
            lon_col = find_first_existing_column(df, ["longitude", "lon", "lng", "x"])
            price_col = find_first_existing_column(
                df,
                ["price_category", "price category", "price", "price_level", "price level"],
            )

            if not rating_col or not lat_col or not lon_col:
                logger.warning("CSV restaurantes sin columnas esperadas de rating/lat/lon.")
                return None

            extra_cols = [c for c in df.columns if c.lower() in ("name", "restaurant", "nombre")]
            if price_col:
                extra_cols.append(price_col)

            df_f = (
                df.filter(F.col(rating_col) >= 3)
                .filter(F.col(lat_col).isNotNull() & F.col(lon_col).isNotNull())
                .select(
                    F.col(rating_col).cast(DoubleType()).alias("rating"),
                    F.col(lat_col).cast(DoubleType()).alias("lat"),
                    F.col(lon_col).cast(DoubleType()).alias("lon"),
                    *[F.col(c) for c in extra_cols],
                )
                .limit(max_points)
            )

            pdf = df_f.toPandas()
        else:
            pdf = pd.read_csv(source)

            cols_lower = {c.lower(): c for c in pdf.columns}
            rating_col = cols_lower.get("rating") or cols_lower.get("stars") or cols_lower.get("score")
            lat_col = cols_lower.get("latitude") or cols_lower.get("lat") or cols_lower.get("y")
            lon_col = cols_lower.get("longitude") or cols_lower.get("lon") or cols_lower.get("lng") or cols_lower.get("x")
            price_col = (
                cols_lower.get("price_category")
                or cols_lower.get("price category")
                or cols_lower.get("price_level")
                or cols_lower.get("price level")
                or cols_lower.get("price")
            )

            if not rating_col or not lat_col or not lon_col:
                logger.warning("CSV restaurantes local sin columnas esperadas de rating/lat/lon.")
                return None

            keep = [rating_col, lat_col, lon_col]
            for n in ["name", "restaurant", "nombre"]:
                if n in cols_lower:
                    keep.append(cols_lower[n])
                    break
            if price_col:
                keep.append(price_col)

            pdf = pdf[keep].copy()
            pdf = pdf.dropna(subset=[rating_col, lat_col, lon_col])
            pdf[rating_col] = pd.to_numeric(pdf[rating_col], errors="coerce")
            pdf[lat_col] = pd.to_numeric(pdf[lat_col], errors="coerce")
            pdf[lon_col] = pd.to_numeric(pdf[lon_col], errors="coerce")
            pdf = pdf.dropna(subset=[rating_col, lat_col, lon_col])
            pdf = pdf[pdf[rating_col] >= 3]
            pdf = pdf.head(max_points)

            rename_map = {rating_col: "rating", lat_col: "lat", lon_col: "lon"}
            if price_col:
                rename_map[price_col] = "price_category"
            pdf = pdf.rename(columns=rename_map)
        if pdf.empty:
            return None

        gdf = gpd.GeoDataFrame(
            pdf,
            geometry=gpd.points_from_xy(pdf["lon"], pdf["lat"]),
            crs="EPSG:4326",
        )

        if gdf_zonas is not None and not gdf_zonas.empty:
            minx, miny, maxx, maxy = gdf_zonas.total_bounds
            gdf = gdf[
                (gdf["lon"] >= minx)
                & (gdf["lon"] <= maxx)
                & (gdf["lat"] >= miny)
                & (gdf["lat"] <= maxy)
            ].copy()
            zones_mask = gdf_zonas[["LocationID", "geometry"]].copy()
            gdf = gpd.sjoin(gdf, zones_mask, how="inner", predicate="within").drop(columns=["index_right"])

        return gdf
    except Exception as e:
        logger.warning("No se pudo cargar restaurantes: %s", str(e).splitlines()[0])
        return None


def cargar_alquiler_por_zona_spark(spark: SparkSession, gdf_zonas: gpd.GeoDataFrame) -> Optional[DataFrame]:
    try:
        rental_path = first_existing_path(RENTALS_LOCAL_CANDIDATES)
        if rental_path is None:
            logger.info("No se encontro dataset local de alquileres; se continua sin esta capa.")
            return None

        pdf = pd.read_csv(rental_path)

        col_map = {c.lower(): c for c in pdf.columns}
        lat_col = col_map.get("latitude") or col_map.get("lat")
        lon_col = col_map.get("longitude") or col_map.get("lon")
        price_col = col_map.get("price")

        if not lat_col or not lon_col or not price_col:
            logger.warning("CSV de alquileres sin columnas esperadas latitude/longitude/price.")
            return None

        pdf = pdf[[lat_col, lon_col, price_col]].copy()
        pdf = pdf.dropna()
        pdf[price_col] = pd.to_numeric(pdf[price_col], errors="coerce")
        pdf = pdf.dropna()
        pdf = pdf[pdf[price_col] > 0]

        if pdf.empty:
            logger.info("Dataset de alquileres sin filas utiles tras limpieza.")
            return None

        gdf_rent = gpd.GeoDataFrame(
            pdf.rename(columns={price_col: "price", lat_col: "lat", lon_col: "lon"}),
            geometry=gpd.points_from_xy(pdf[lon_col], pdf[lat_col]),
            crs="EPSG:4326",
        )

        zonas = gdf_zonas[["LocationID", "geometry"]].copy()
        joined = gpd.sjoin(gdf_rent, zonas, how="inner", predicate="within")
        if joined.empty:
            logger.info("No hubo matches espaciales entre alquileres y taxi zones.")
            return None

        rent_agg = (
            joined.groupby("LocationID", as_index=False)
            .agg(
                alquiler_medio=("price", "mean"),
                alquiler_mediano=("price", "median"),
                alquiler_count=("price", "size"),
            )
        )

        return spark.createDataFrame(rent_agg)
    except Exception as e:
        logger.warning("No se pudo cargar/agregar alquileres: %s", str(e).splitlines()[0])
        return None


def incorporar_alquiler_metricas(metricas: DataFrame, alquiler_df: Optional[DataFrame]) -> DataFrame:
    if alquiler_df is None:
        return metricas

    return metricas.join(alquiler_df, on="LocationID", how="left").fillna(
        {
            "alquiler_medio": 0.0,
            "alquiler_mediano": 0.0,
            "alquiler_count": 0,
        }
    )


def crear_mapa(gdf_merged: gpd.GeoDataFrame, usar_cluster: bool = True) -> folium.Map:
    nyc_center = [40.7128, -74.0060]
    m = folium.Map(location=nyc_center, zoom_start=10, tiles="cartodbpositron", control_scale=True)

    min_val = float(gdf_merged["poder_adquisitivo_0_100"].min())
    max_val = float(gdf_merged["poder_adquisitivo_0_100"].max())

    colormap = cm.LinearColormap(
        colors=["#f7fcf5", "#a1d99b", "#31a354", "#006d2c"],
        vmin=min_val,
        vmax=max_val,
        caption="Mapa de intensidad",
    )

    def style_power(feature):
        value = feature["properties"].get("poder_adquisitivo_0_100", 0.0)
        fill = colormap(value) if value is not None else "#d9d9d9"
        return {
            "fillColor": fill,
            "color": "#333333",
            "weight": 0.6,
            "fillOpacity": 0.75,
        }

    tooltip_fields = [
        "zone",
        "borough",
        "LocationID",
        "volumen_viajes",
        "poder_adquisitivo_0_100",
    ]
    tooltip_fields = [c for c in tooltip_fields if c in gdf_merged.columns]
    aliases = [
        "Zone",
        "Borough",
        "LocationID",
        "Cantidad de viajes",
        "Poder adquisitivo",
    ][: len(tooltip_fields)]

    layer_power = folium.GeoJson(
        gdf_merged,
        name="Poder adquisitivo",
        overlay=True,
        style_function=style_power,
        highlight_function=lambda _: {"weight": 2, "fillOpacity": 0.95},
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=aliases, sticky=True),
    )
    layer_power.add_to(m)
    colormap.add_to(m)

    layer_cluster = None
    if usar_cluster and "cluster" in gdf_merged.columns:
        cluster_cmap = cm.LinearColormap(
            colors=["#005a32", "#238b45", "#41ab5d", "#74c476", "#a1d99b", "#e5f5e0"],
            vmin=0,
            vmax=5,
        )

        def style_cluster(feature):
            c = feature["properties"].get("cluster", -1)
            if c is None or int(c) < 0:
                return {
                    "fillColor": "#d9d9d9",
                    "color": "#8c8c8c",
                    "weight": 0.4,
                    "fillOpacity": 0.35,
                }
            color = cluster_cmap(int(c))
            return {
                "fillColor": color,
                "color": "#222222",
                "weight": 0.6,
                "fillOpacity": 0.80,
            }

        layer_cluster = folium.GeoJson(
            gdf_merged,
            name="Cluster KMeans",
            overlay=True,
            style_function=style_cluster,
            highlight_function=lambda _: {"weight": 2, "fillOpacity": 0.9},
            tooltip=folium.GeoJsonTooltip(
                fields=[c for c in ["zone", "borough", "LocationID", "cluster_label"] if c in gdf_merged.columns],
                aliases=["Zone", "Borough", "LocationID", "Cluster"],
                sticky=True,
            ),
            show=False,
        )
        layer_cluster.add_to(m)

    plugins.MiniMap(toggle_display=True).add_to(m)
    plugins.Fullscreen(position="topleft").add_to(m)

    # Forzamos exclusividad entre capas tematicas sin tocar el basemap,
    # y mostramos la barra solo cuando esta activa "Poder adquisitivo".
    cluster_layer_js = layer_cluster.get_name() if layer_cluster else "null"
    script = f"""
    (function() {{
        var mapName = "{m.get_name()}";
        var powerName = "{layer_power.get_name()}";
        var clusterName = {('"' + layer_cluster.get_name() + '"') if layer_cluster else 'null'};

        function getLegendEl() {{
            var legends = document.getElementsByClassName('legend');
            if (!legends || legends.length === 0) return null;
            return legends[0];
        }}

        function sameLayer(a, b) {{
            return !!(a && b && a._leaflet_id === b._leaflet_id);
        }}

        function initWhenReady() {{
            var mapObj = window[mapName];
            var powerLayer = window[powerName];
            var clusterLayer = clusterName ? window[clusterName] : null;

            if (!mapObj || !powerLayer || (clusterName && !clusterLayer)) {{
                return false;
            }}

            function isPowerCheckedInControl() {{
                var labels = document.querySelectorAll('.leaflet-control-layers-overlays label');
                for (var i = 0; i < labels.length; i++) {{
                    var label = labels[i];
                    var input = label.querySelector('input');
                    var text = (label.textContent || '').trim();
                    if (text.indexOf('Poder adquisitivo') !== -1) {{
                        return !!(input && input.checked);
                    }}
                }}
                return mapObj.hasLayer(powerLayer);
            }}

            function syncLegend() {{
                var legend = getLegendEl();
                if (!legend) return;
                legend.style.display = isPowerCheckedInControl() ? 'block' : 'none';
            }}

            function setupThemeRadios() {{
                var labels = document.querySelectorAll('.leaflet-control-layers-overlays label');
                var powerInput = null;
                var clusterInput = null;

                for (var i = 0; i < labels.length; i++) {{
                    var label = labels[i];
                    var input = label.querySelector('input[type="checkbox"]');
                    var text = (label.textContent || '').trim();
                    if (!input) continue;

                    if (text.indexOf('Poder adquisitivo') !== -1) {{
                        input.type = 'radio';
                        input.name = 'tema-mapa';
                        powerInput = input;
                    }} else if (text.indexOf('Cluster KMeans') !== -1) {{
                        input.type = 'radio';
                        input.name = 'tema-mapa';
                        clusterInput = input;
                    }}
                }}

                if (powerInput && clusterInput) {{
                    if (!mapObj.hasLayer(powerLayer) && !mapObj.hasLayer(clusterLayer)) {{
                        mapObj.addLayer(powerLayer);
                    }}
                }}
            }}

            mapObj.on('overlayadd', function(e) {{
                if (clusterLayer && sameLayer(e.layer, powerLayer) && mapObj.hasLayer(clusterLayer)) {{
                    mapObj.removeLayer(clusterLayer);
                }}
                if (clusterLayer && sameLayer(e.layer, clusterLayer) && mapObj.hasLayer(powerLayer)) {{
                    mapObj.removeLayer(powerLayer);
                }}
                syncLegend();
            }});

            mapObj.on('overlayremove', function() {{
                // Evita que el usuario deje sin capa tematica activa.
                if (!mapObj.hasLayer(powerLayer) && (!clusterLayer || !mapObj.hasLayer(clusterLayer))) {{
                    mapObj.addLayer(powerLayer);
                }}
                syncLegend();
            }});

            setupThemeRadios();
            setTimeout(syncLegend, 0);
            return true;
        }}

        var tries = 0;
        var timer = setInterval(function() {{
            if (initWhenReady() || tries > 40) {{
                clearInterval(timer);
            }}
            tries += 1;
        }}, 50);
    }})();
    """
    m.get_root().script.add_child(Element(script))

    title_html = """
    <div style="
        position: fixed;
        top: 12px;
        left: 60px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.90);
        padding: 8px 12px;
        border: 1px solid #cccccc;
        border-radius: 6px;
        font-size: 18px;
        font-weight: 700;
    ">
        NYC GeoCore
    </div>
    """
    m.get_root().html.add_child(Element(title_html))

    blue_point_legend = """
    <div style="
        position: fixed;
        bottom: 16px;
        left: 12px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.90);
        padding: 8px 10px;
        border: 1px solid #cccccc;
        border-radius: 5px;
        font-size: 12px;
        line-height: 1.2;
    ">
        <div style="font-weight:700; margin-bottom:4px;">Restaurantes</div>
        <div style="color:#245b87; margin-bottom:5px;">Azul mas intenso = mayor rating</div>

        <div style="display:flex; align-items:center; gap:6px; margin-bottom:2px;">
            <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
                <polygon points="6,1 11,11 1,11" fill="#5dade2"></polygon>
            </svg>
            <span>$ = Economico</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px; margin-bottom:2px;">
            <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
                <rect x="1.5" y="1.5" width="9" height="9" fill="#5dade2"></rect>
            </svg>
            <span>$$ = Medio</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px; margin-bottom:2px;">
            <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
                <polygon points="6,1 10.8,4.7 9,10.8 3,10.8 1.2,4.7" fill="#5dade2"></polygon>
            </svg>
            <span>$$$ = Alto</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
            <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
                <polygon points="6,1 10,3.1 10,8.9 6,11 2,8.9 2,3.1" fill="#5dade2"></polygon>
            </svg>
            <span>$$$$ = Muy alto</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(Element(blue_point_legend))

    return m


def agregar_capa_restaurantes(m: folium.Map, gdf_rest: Optional[gpd.GeoDataFrame]) -> folium.Map:
    if gdf_rest is None or gdf_rest.empty:
        return m

    def normalizar_categoria_precio(raw_price) -> str:
        if raw_price is None or pd.isna(raw_price):
            return "Sin dato"

        txt = str(raw_price).strip()
        if not txt:
            return "Sin dato"

        try:
            level = int(float(txt))
            level = min(max(level, 1), 4)
            return "$" * level
        except Exception:
            pass

        dollar_count = txt.count("$")
        if dollar_count > 0:
            return "$" * min(dollar_count, 4)

        txt_low = txt.lower()
        if txt_low in {"low", "cheap", "economico", "budget"}:
            return "$"
        if txt_low in {"medium", "moderate", "medio"}:
            return "$$"
        if txt_low in {"high", "alto", "expensive"}:
            return "$$$"
        if txt_low in {"luxury", "premium", "muy alto"}:
            return "$$$$"

        return "Sin dato"

    fg = folium.FeatureGroup(name="Restaurantes", show=False)
    for _, row in gdf_rest.iterrows():
        rating = row.get("rating", None)
        price_cat = row.get("price_category", None)

        # Paleta mas clara: color por rating, tamano por categoria de precio.
        if rating is None:
            marker_color = "#5dade2"
        elif rating >= 4.7:
            marker_color = "#1f78b4"
        elif rating >= 4.4:
            marker_color = "#5dade2"
        else:
            marker_color = "#a6cee3"

        categoria_precio = normalizar_categoria_precio(price_cat)
        marker_sides = {
            "$": 3,
            "$$": 4,
            "$$$": 5,
            "$$$$": 6,
            "Sin dato": 24,
        }.get(categoria_precio, 24)

        marker_radius = 6.0
        marker_rotation = 45 if categoria_precio == "$$" else 0

        popup_parts = [f"Rating: {row.get('rating', 'N/A')}"]
        popup_parts.append(f"Categoria precio: {categoria_precio}")
        if "name" in row and row["name"] is not None:
            popup_parts.insert(0, f"Nombre: {row['name']}")
        elif "restaurant" in row and row["restaurant"] is not None:
            popup_parts.insert(0, f"Restaurante: {row['restaurant']}")
        elif "nombre" in row and row["nombre"] is not None:
            popup_parts.insert(0, f"Nombre: {row['nombre']}")

        folium.RegularPolygonMarker(
            location=[row["lat"], row["lon"]],
            number_of_sides=marker_sides,
            radius=marker_radius,
            rotation=marker_rotation,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.92,
            weight=1.0,
            popup="<br>".join(popup_parts),
        ).add_to(fg)

    fg.add_to(m)
    return m


# -----------------------------------------------------------------------------
# Save outputs
# -----------------------------------------------------------------------------
def guardar_outputs(df_final: DataFrame, mapa: folium.Map) -> None:
    output_data_dir_abs = resolve_project_path(str(OUTPUT_DATA_DIR))
    output_map_dir_abs = resolve_project_path(str(OUTPUT_MAP_DIR))
    output_parquet_abs = resolve_project_path(str(OUTPUT_PARQUET))
    output_html_abs = resolve_project_path(str(OUTPUT_HTML))

    output_data_dir_abs.mkdir(parents=True, exist_ok=True)
    output_map_dir_abs.mkdir(parents=True, exist_ok=True)

    # Si existe como carpeta (resto de una escritura Spark anterior), la borramos.
    if output_parquet_abs.exists():
        if output_parquet_abs.is_dir():
            shutil.rmtree(output_parquet_abs)
        else:
            output_parquet_abs.unlink()

    # Guardamos usando Pandas para evitar el error de "NativeIO" en Windows
  
    pdf_final = df_final.toPandas()
    tmp_parquet = output_parquet_abs.with_suffix(".tmp.parquet")
    if tmp_parquet.exists():
        tmp_parquet.unlink()

    pdf_final.to_parquet(str(tmp_parquet), index=False, engine="pyarrow")
    tmp_parquet.replace(output_parquet_abs)
    logger.info("Dataset agregado guardado localmente en: %s", output_parquet_abs)

    mapa.save(str(output_html_abs))
    logger.info("Mapa interactivo guardado en: %s", output_html_abs)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    spark = None
    try:
        logger.info("=== Inicio pipeline NYC Taxi + FHV ===")
        spark = crear_spark_session()

        taxi_raw = cargar_parquet_con_fallback(spark, "taxi")
        fhv_raw = cargar_parquet_con_fallback(spark, "fhv")
        gdf_zonas = cargar_zonas_gdf()

        taxi_agg = preparar_taxi(taxi_raw)
        fhv_agg = preparar_fhv(fhv_raw)
        metricas = combinar_metricas(taxi_agg, fhv_agg)

        alquiler_agg = cargar_alquiler_por_zona_spark(spark, gdf_zonas)
        metricas = incorporar_alquiler_metricas(metricas, alquiler_agg)

        metricas = calcular_indice_poder_adquisitivo(metricas)

        # Bonus clustering
        metricas, model = aplicar_kmeans(metricas, k=6)
        logger.info("KMeans entrenado. K=%s", model.getK())

        # Solo al final pasamos a pandas/geopandas para mapa
        pdf = metricas.toPandas()
        gdf_metricas = gpd.GeoDataFrame(pdf)

        gdf_merged = gdf_zonas.merge(gdf_metricas, on="LocationID", how="left")

        # Fill para zonas sin viajes
        fill_cols = [
            "tip_amount_avg", "passenger_count_avg", "taxi_trip_count", "fhv_trip_count", 
            "volumen_viajes", "alquiler_medio", "alquiler_mediano", "alquiler_count",
            "tip_norm", "passenger_norm", "volumen_norm", "alquiler_norm",
            "poder_adquisitivo", "poder_adquisitivo_0_100"
        ]
        for c in fill_cols:
            if c in gdf_merged.columns:
                gdf_merged[c] = gdf_merged[c].fillna(0)

        if "cluster" in gdf_merged.columns:
            gdf_merged["cluster"] = gdf_merged["cluster"].fillna(-1)
        if "cluster_label" in gdf_merged.columns:
            gdf_merged["cluster_label"] = gdf_merged["cluster_label"].fillna("Sin datos")

        m = crear_mapa(gdf_merged, usar_cluster=True)

        # Capa extra opcional restaurantes
        gdf_rest = cargar_restaurantes_filtrados(spark, gdf_zonas=gdf_zonas)
        m = agregar_capa_restaurantes(m, gdf_rest)

        # LayerControl al final para incluir tambien restaurantes.
        folium.LayerControl(collapsed=False).add_to(m)

        # Guardar parquet final y mapa HTML
        guardar_outputs(metricas, m)

        logger.info("=== Pipeline finalizado OK ===")
    except Exception as e:
        logger.exception("Error en el pipeline: %s", e)
        raise
    finally:
        if spark is not None:
            try:
                spark.stop()
                logger.info("Spark detenido.")
            except Exception as stop_err:
                logger.warning("No se pudo detener Spark limpiamente: %s", stop_err)


if __name__ == "__main__":
    main()