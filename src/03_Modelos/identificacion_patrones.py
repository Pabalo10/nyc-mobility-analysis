import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import geopandas as gpd
import folium
from folium.plugins import TimestampedGeoJson
from sklearn.cluster import KMeans
import numpy as np
import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from dotenv import load_dotenv
import platform
import sys

load_dotenv()
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["HOSTNAME"] = "localhost"
if platform.system() == "Windows":
    os.environ["HADOOP_HOME"] = r"C:\hadoop"
    
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# --------------------------------------------------
# MINIO CONFIG
# --------------------------------------------------
MINIO_CONFIG = {
    "endpoint": os.getenv("MINIO_ENDPOINT", "https://minio.fdi.ucm.es"),
    "access_key": os.getenv("MINIO_ACCESS_KEY", ""),
    "secret_key": os.getenv("MINIO_SECRET_KEY", ""),
}

S3_PATH = "s3a://pd2/taxomanos/limpios/nyc_taxi_clean.parquet"

# --------------------------------------------------
# RUTAS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "datos" / "limpios"
OUT_DIR = Path(__file__).resolve().parents[1] / "Visualizacion" / "Patrones_Demanda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# SPARK
# --------------------------------------------------
def crear_spark():

    spark = (SparkSession.builder
        .appName("TaxiDemandAnalysis")
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.jars.packages",
                "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262")
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_CONFIG["endpoint"])
        .config("spark.hadoop.fs.s3a.access.key", MINIO_CONFIG["access_key"])
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_CONFIG["secret_key"])
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

# --------------------------------------------------
# 1. CARGA DE DATOS
# --------------------------------------------------
def load_data(DATA_DIR, spark):
    print("Intentando cargar datos desde MinIO...")
    try:
        df = spark.read.parquet(S3_PATH)
        print("✓ Datos cargados desde MinIO")
        return df
    except Exception as e:
        print(f"⚠ Fallo en MinIO: {str(e).splitlines()[0]}")

    try:
        print("Cargando desde local...")
        df = spark.read.parquet(str(DATA_DIR / "nyc_taxi_clean.parquet"))
        print("✓ Datos cargados desde local")
        return df
    except Exception as e:
        print("❌ Error crítico: no se pudo cargar el dataset")
        raise e

# --------------------------------------------------
# 2. VARIABLES TEMPORALES
# --------------------------------------------------
def add_time_variables(df):
    return (
        df
        .withColumn("hora", F.col("pickup_hour"))
        .withColumn("dia_semana", F.col("pickup_weekday"))
        .withColumn("fecha", F.to_date("tpep_pickup_datetime"))
    )

# --------------------------------------------------
# 3. DEMANDA DIARIA
# --------------------------------------------------
def build_daily_demand(df):
    return (
        df
        .groupBy("fecha", "pulocationid", "hora")
        .agg(F.count("*").alias("demanda"))
    )

# --------------------------------------------------
# 4. PATRÓN HORARIO
# --------------------------------------------------
def build_hourly_pattern(demanda_diaria):
    return (
        demanda_diaria
        .groupBy("pulocationid", "hora")
        .agg(F.avg("demanda").alias("demanda"))
    )

# --------------------------------------------------
# 5. CLASIFICACIÓN
# --------------------------------------------------
def classify_demand(demanda):
    thresholds = (
        demanda
        .groupBy("pulocationid")
        .agg(
            F.expr("percentile_approx(demanda, 0.33)").alias("Q1"),
            F.expr("percentile_approx(demanda, 0.66)").alias("Q3")
        )
    )

    return (
        demanda
        .join(thresholds, on="pulocationid", how="left")
        .withColumn(
            "nivel_demanda",
            F.when(F.col("demanda") <= F.col("Q1"), "baja")
             .when(F.col("demanda") >= F.col("Q3"), "alta")
             .otherwise("media")
        )
    )

# --------------------------------------------------
# 6. HEATMAP
# --------------------------------------------------
def plot_main_heatmap(demanda, OUT_DIR):
    df = demanda.toPandas()

    pivot = df.pivot_table(
        index="pulocationid",
        columns="hora",
        values="demanda",
        aggfunc="mean"
    )

    plt.figure(figsize=(12,8))
    sns.heatmap(pivot, cmap="coolwarm")
    plt.title("Patrón de demanda por zona y hora")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "heatmap_patrones.png")
    plt.close()

# --------------------------------------------------
# 7. TOP ZONAS
# --------------------------------------------------
def plot_top_zones(demanda, OUT_DIR):
    top_zonas = (
        demanda
        .groupBy("pulocationid")
        .agg(F.avg("demanda").alias("media"))
        .orderBy(F.desc("media"))
        .limit(5)
        .select("pulocationid")
        .toPandas()["pulocationid"]
        .tolist()
    )

    df = (
        demanda
        .filter(F.col("pulocationid").isin(top_zonas))
        .toPandas()
    )

    plt.figure(figsize=(12,6))

    for zona in top_zonas:
        curva = df[df["pulocationid"] == zona].sort_values("hora")
        plt.plot(curva["hora"], curva["demanda"], label=f"Zona {zona}")

    plt.legend()
    plt.title("Curvas de demanda (Top zonas)")
    plt.xlabel("Hora")
    plt.ylabel("Demanda media")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "curvas_demanda.png")
    plt.close()

# --------------------------------------------------
def plot_boxplot(demanda, OUT_DIR):
    df = demanda.toPandas()

    plt.figure(figsize=(12,6))
    sns.boxplot(x="hora", y="demanda", data=df)
    plt.title("Distribución de demanda por hora")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "boxplot_demanda.png")
    plt.close()

# --------------------------------------------------
# SEMANAL
# --------------------------------------------------
def build_weekly_demand(df):
    # 1. demanda por día real
    demanda_diaria = (
        df
        .groupBy("fecha", "dia_semana", "hora")
        .agg(F.count("*").alias("demanda"))
    )

    # 2. media por día de la semana y hora
    demanda_semana = (
        demanda_diaria
        .groupBy("dia_semana", "hora")
        .agg(F.avg("demanda").alias("demanda"))
    )

    return demanda_semana
def plot_weekly_curves(demanda_semana, OUT_DIR):
    df = demanda_semana.toPandas()

    mapa = {0:"Lunes",1:"Martes",2:"Miércoles",3:"Jueves",4:"Viernes",5:"Sábado",6:"Domingo"}
    df["dia_semana"] = df["dia_semana"].map(mapa)

    orden = list(mapa.values())

    plt.figure(figsize=(12,6))

    for dia in orden:
        sub = df[df["dia_semana"] == dia].sort_values("hora")
        plt.plot(sub["hora"], sub["demanda"], label=dia)

    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "curvas_dias_semana.png")
    plt.close()

def plot_weekly_heatmap(demanda_semana, OUT_DIR):
    df = demanda_semana.toPandas()

    mapa = {0:"Lunes",1:"Martes",2:"Miércoles",3:"Jueves",4:"Viernes",5:"Sábado",6:"Domingo"}
    df["dia_semana"] = df["dia_semana"].map(mapa)

    orden = list(mapa.values())

    pivot = df.pivot_table(
        index="dia_semana",
        columns="hora",
        values="demanda",
        aggfunc="mean"
    ).reindex(orden)

    plt.figure(figsize=(12,6))
    sns.heatmap(pivot, cmap="coolwarm")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "heatmap_dias_semana.png")
    plt.close()
    
def plot_global_demand_distribution(demanda, OUT_DIR):
    print("[9] Distribución global por nivel de demanda...")

    df = (
        demanda
        .groupBy("hora", "nivel_demanda")
        .agg(F.count("*").alias("count"))
    )

    total = df.groupBy("hora").agg(F.sum("count").alias("total"))

    df = (
        df.join(total, "hora")
        .withColumn("ratio", F.col("count") / F.col("total"))
        .toPandas()
    )

    pivot = df.pivot(index="hora", columns="nivel_demanda", values="ratio").fillna(0)
    pivot = pivot[["baja", "media", "alta"]]  # orden fijo

    plt.figure(figsize=(12,6))
    plt.stackplot(pivot.index, pivot.T, labels=pivot.columns)

    plt.legend(loc="upper left")
    plt.title("Distribución global de niveles de demanda")
    plt.xlabel("Hora")
    plt.ylabel("Proporción")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "stacked_demanda_global.png")
    plt.close()

# --------------------------------------------------
# GRID
# --------------------------------------------------
def build_full_grid(demanda, zones):
    spark = demanda.sparkSession

    zonas_ids = zones["LocationID"].unique().tolist()

    horas_df = spark.createDataFrame([(i,) for i in range(24)], ["hora"])
    zonas_df = spark.createDataFrame([(int(z),) for z in zonas_ids], ["pulocationid"])

    grid = zonas_df.crossJoin(horas_df)

    return (
        grid
        .join(demanda, on=["pulocationid", "hora"], how="left")
        .fillna({"demanda": 0})
    )
    
def map_dominant_demand(demanda, DATA_DIR, OUT_DIR):
    print("[10] Mapa de demanda dominante por zona...")

    from pyspark.sql.window import Window

    modo = (
        demanda
        .groupBy("pulocationid", "nivel_demanda")
        .agg(F.count("*").alias("freq"))
    )

    w = Window.partitionBy("pulocationid").orderBy(F.desc("freq"))

    modo = (
        modo
        .withColumn("rank", F.row_number().over(w))
        .filter(F.col("rank") == 1)
        .select("pulocationid", "nivel_demanda")
        .toPandas()
    )

    zones = gpd.read_file(DATA_DIR / "taxi_zones" / "taxi_zones.shp").to_crs(epsg=4326)

    zones = zones.merge(
        modo,
        left_on="LocationID",
        right_on="pulocationid",
        how="left"
    )

    color_map = {
        "baja": "#2ca25f",
        "media": "#feb24c",
        "alta": "#de2d26"
    }

    mapa = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles="CartoDB positron")

    for _, row in zones.iterrows():
        if pd.isna(row["nivel_demanda"]):
            continue

        folium.GeoJson(
            row["geometry"],
            style_function=lambda x, c=row["nivel_demanda"]: {
                "fillColor": color_map.get(c, "#999"),
                "color": "black",
                "weight": 0.4,
                "fillOpacity": 0.7
            },
            tooltip=f"{row.get('zone','')} → {row['nivel_demanda']}"
        ).add_to(mapa)

    legend_html = """
    <div style="
    position: fixed; 
    bottom: 40px; left: 40px; width: 160px; height: 120px; 
    background-color: white; 
    border:2px solid grey; z-index:9999; font-size:14px;
    padding: 10px;
    ">
    <b>Demanda</b><br>
    <i style="background:#2ca25f;width:10px;height:10px;display:inline-block;"></i> Baja<br>
    <i style="background:#feb24c;width:10px;height:10px;display:inline-block;"></i> Media<br>
    <i style="background:#de2d26;width:10px;height:10px;display:inline-block;"></i> Alta
    </div>
    """

    mapa.get_root().html.add_child(folium.Element(legend_html))
    mapa.save(OUT_DIR / "mapa_demanda_dominante.html")
    


# --------------------------------------------------
# MAPAS
# --------------------------------------------------
def add_legend(mapa, title="Demanda"):
    legend_html = f"""<div style="position: fixed; bottom: 40px; right: 40px;
    width: 160px; height: 110px; z-index:9999; font-size:14px;
    background-color:white; padding:10px; border:2px solid grey;
    border-radius:8px;">
    <b>{title}</b><br>
    <i style="background:#2ca25f;width:10px;height:10px;display:inline-block;"></i> Baja<br>
    <i style="background:#feb24c;width:10px;height:10px;display:inline-block;"></i> Media<br>
    <i style="background:#de2d26;width:10px;height:10px;display:inline-block;"></i> Alta
    </div>"""
    mapa.get_root().html.add_child(folium.Element(legend_html))
    


def map_global(demanda, DATA_DIR, OUT_DIR):
    print("Generando mapa GLOBAL...")

    zones = gpd.read_file(DATA_DIR / "taxi_zones" / "taxi_zones.shp")
    zones = zones.to_crs(epsg=4326)
    zones["LocationID"] = zones["LocationID"].astype(int)

    # ---------------------------
    # GRID
    # ---------------------------
    demanda_full = build_full_grid(demanda, zones)

    # ---------------------------
    # CUARTILES
    # ---------------------------
    quantiles = demanda_full.approxQuantile("demanda", [0.33, 0.66], 0.01)
    q1, q3 = quantiles[0], quantiles[1]

    # ---------------------------
    # CLASIFICACIÓN
    # ---------------------------
    demanda_full = demanda_full.withColumn(
        "nivel",
        F.when(F.col("demanda") <= q1, "baja")
         .when(F.col("demanda") >= q3, "alta")
         .otherwise("media")
    )

    demanda_full = demanda_full.toPandas()

    # ---------------------------
    # COLORES
    # ---------------------------
    color_map = {
        "baja": "#2ca25f",
        "media": "#feb24c",
        "alta": "#de2d26"
    }

    # ---------------------------
    # FEATURES
    # ---------------------------
    features = []

    for _, zona in zones.iterrows():
        subset = demanda_full[demanda_full["pulocationid"] == zona["LocationID"]]

        for _, row in subset.iterrows():
            features.append({
                "type": "Feature",
                "geometry": zona["geometry"].__geo_interface__,
                "properties": {
                    "times": [f"2023-01-01T{int(row['hora']):02d}:00:00Z"],
                    "style": {
                        "fillColor": color_map[row["nivel"]],
                        "color": "black",
                        "fillOpacity": 1,
                        "weight": 0.5
                    },
                    "tooltip": f"{zona.get('zone','')} → {row['nivel']}"
                }
            })

    geojson = {"type": "FeatureCollection", "features": features}

    mapa = folium.Map(
        location=[40.7128, -74.0060],
        zoom_start=11,
        tiles="CartoDB positron"
    )

    TimestampedGeoJson(geojson, period="PT1H", duration="PT1H").add_to(mapa)

    add_legend(mapa, "Demanda global")

    mapa.save(OUT_DIR / "mapa_global.html")

def map_local(demanda, DATA_DIR, OUT_DIR):
    print("Generando mapa POR ZONA...")

    zones = gpd.read_file(DATA_DIR / "taxi_zones" / "taxi_zones.shp")
    zones = zones.to_crs(epsg=4326)
    zones["LocationID"] = zones["LocationID"].astype(int)

    demanda_full = build_full_grid(demanda, zones)

    # ---------------------------
    # CUARTILES POR ZONA
    # ---------------------------
    quantiles = (
    demanda_full
    .groupBy("pulocationid")
    .agg(
        F.expr("percentile_approx(demanda, 0.33)").alias("q1_local"),
        F.expr("percentile_approx(demanda, 0.66)").alias("q3_local")
    )
)

    demanda_local = demanda_full.join(quantiles, on="pulocationid")
    # ---------------------------
    # CLASIFICACIÓN
    # ---------------------------
    demanda_local = demanda_local.withColumn(
    "nivel",
    F.when(F.col("demanda") <= F.col("q1_local"), "baja")
     .when(F.col("demanda") >= F.col("q3_local"), "alta")
     .otherwise("media")
)

    demanda_local = demanda_local.toPandas()

    # ---------------------------
    # COLORES
    # ---------------------------
    color_map = {
        "baja": "#2ca25f",
        "media": "#feb24c",
        "alta": "#de2d26"
    }

    # ---------------------------
    # FEATURES
    # ---------------------------
    features = []

    for _, zona in zones.iterrows():
        subset = demanda_local[demanda_local["pulocationid"] == zona["LocationID"]]

        for _, row in subset.iterrows():
            features.append({
                "type": "Feature",
                "geometry": zona["geometry"].__geo_interface__,
                "properties": {
                    "times": [f"2023-01-01T{int(row['hora']):02d}:00:00Z"],
                    "style": {
                        "fillColor": color_map[row["nivel"]],
                        "color": "black",
                        "fillOpacity": 1,
                        "weight": 0.5
                    },
                    "tooltip": f"{zona.get('zone','')} → {row['nivel']}"
                }
            })

    geojson = {"type": "FeatureCollection", "features": features}

    mapa = folium.Map(
        location=[40.7128, -74.0060],
        zoom_start=11,
        tiles="CartoDB positron"
    )

    TimestampedGeoJson(geojson, period="PT1H", duration="PT1H").add_to(mapa)

    add_legend(mapa, "Demanda por zona")

    mapa.save(OUT_DIR / "mapa_por_zona.html")

def clustering_analysis(demanda, DATA_DIR, OUT_DIR, k=3):
    print("Iniciando clustering...")

    # --------------------------------------------------
    # 1. SPARK → PANDAS
    # --------------------------------------------------
    demanda = demanda.toPandas()

    # --------------------------------------------------
    # 2. MATRIZ
    # --------------------------------------------------
    matriz = demanda.pivot(
        index="pulocationid",
        columns="hora",
        values="demanda"
    ).fillna(0)

    matriz = matriz.reindex(sorted(matriz.columns), axis=1)

    # --------------------------------------------------
    # 3. NORMALIZACIÓN
    # --------------------------------------------------
    X = np.log1p(matriz)

    X = X.sub(X.mean(axis=1), axis=0)
    X = X.div(X.std(axis=1), axis=0).fillna(0)

    # --------------------------------------------------
    # 4. KMEANS
    # --------------------------------------------------
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    matriz["cluster"] = kmeans.fit_predict(X)

    # --------------------------------------------------
    # 5. INTERPRETACIÓN CLUSTERS
    # --------------------------------------------------
    cluster_labels = {}

    for c in range(k):
        subset = matriz[matriz["cluster"] == c].drop(columns="cluster")

        if subset.empty:
            cluster_labels[c] = "Sin datos"
            continue

        media = subset.mean()

        top_horas = media.sort_values(ascending=False).head(2).index.tolist()
        pico = top_horas[0]

        if 7 <= pico <= 10:
            label = f"Mañana (pico {pico}h)"
        elif 17 <= pico <= 20:
            label = f"Tarde / commuting (pico {pico}h)"
        elif pico >= 21 or pico <= 3:
            label = f"Nocturno (pico {pico}h)"
        else:
            label = f"Mixto (pico {pico}h)"

        cluster_labels[c] = label

    # --------------------------------------------------
    # 6. CURVAS
    # --------------------------------------------------
    plt.figure(figsize=(12,6))

    for c in range(k):
        subset = matriz[matriz["cluster"] == c].drop(columns="cluster")

        if subset.empty:
            continue

        media = subset.mean()

        plt.plot(media.index, media.values, label=cluster_labels[c])

    plt.legend()
    plt.title("Patrones por cluster")
    plt.xlabel("Hora")
    plt.ylabel("Demanda normalizada")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "clusters_curvas.png")
    plt.close()

    # --------------------------------------------------
    # 7. MAPA
    # --------------------------------------------------
    print("Generando mapa de clusters...")

    zones = gpd.read_file(DATA_DIR / "taxi_zones" / "taxi_zones.shp")
    zones = zones.to_crs(epsg=4326)

    clusters_df = matriz["cluster"].reset_index()

    zones = zones.merge(
        clusters_df,
        left_on="LocationID",
        right_on="pulocationid",
        how="left"
    )

    cluster_colors = {
        0: "#1f77b4",
        1: "#2ca02c",
        2: "#d62728",
        3: "#9467bd",
        4: "#ff7f0e"
    }

    mapa = folium.Map(
        location=[40.7128, -74.0060],
        zoom_start=11,
        tiles="CartoDB positron"
    )

    for _, row in zones.iterrows():
        if pd.isna(row["cluster"]):
            continue

        cluster = int(row["cluster"])
        label = cluster_labels[cluster]

        folium.GeoJson(
            row["geometry"],
            style_function=lambda x, c=cluster: {
                "fillColor": cluster_colors.get(c, "#999999"),
                "color": "black",
                "weight": 0.4,
                "fillOpacity": 0.7
            },
            tooltip=f"{row.get('zone','Zona')} ({row.get('borough','')})<br>{label}"
        ).add_to(mapa)

    # --------------------------------------------------
    # 8. LEYENDA
    # --------------------------------------------------
    legend_html = """
    <div style="
    position: absolute;
    z-index: 9999;
    bottom: 40px;
    right: 40px;
    width: 260px;
    background-color:white;
    padding:10px;
    border:2px solid grey;
    border-radius:8px;
    font-size:14px;
    ">
    <b>Tipos de zona</b><br>
    """

    for c, label in cluster_labels.items():
        color = cluster_colors.get(c, "#999999")
        legend_html += f"""
        <i style="background:{color};width:10px;height:10px;display:inline-block;"></i>
        {label}<br>
        """

    legend_html += "</div>"

    mapa.get_root().html.add_child(folium.Element(legend_html))

    mapa.save(OUT_DIR / "mapa_clusters.html")

    print("Clustering completado")

# --------------------------------------------------
def main():
    print("===================================")
    print("INICIANDO PIPELINE DE ANÁLISIS")
    print("===================================")

    spark = crear_spark()

    # --------------------------------------------------
    # 1. CARGA Y PREPARACIÓN
    # --------------------------------------------------
    print("\n[1] Cargando datos...")
    df = load_data(DATA_DIR, spark)

    print("[2] Añadiendo variables temporales...")
    df = add_time_variables(df)

    # --------------------------------------------------
    # 2. DEMANDA BASE
    # --------------------------------------------------
    print("\n[3] Construyendo demanda diaria...")
    demanda_diaria = build_daily_demand(df)

    print("[4] Calculando patrón horario medio...")
    demanda = build_hourly_pattern(demanda_diaria)

    print("[5] Clasificando demanda por zona...")
    demanda = classify_demand(demanda)

    # --------------------------------------------------
    # 3. VISUALIZACIONES BÁSICAS
    # --------------------------------------------------
    print("\n[6] Generando heatmap principal...")
    plot_main_heatmap(demanda, OUT_DIR)

    print("[7] Generando curvas de zonas top...")
    plot_top_zones(demanda, OUT_DIR)

    print("[8] Generando boxplot...")
    plot_boxplot(demanda, OUT_DIR)

    plot_global_demand_distribution(demanda, OUT_DIR)

    map_dominant_demand(demanda, DATA_DIR, OUT_DIR)

    # --------------------------------------------------
    # 4. ANÁLISIS SEMANAL
    # --------------------------------------------------
    print("\n[11] Analizando demanda por día de la semana...")
    demanda_semana = build_weekly_demand(df)

    print("[12] Generando curvas semanales...")
    plot_weekly_curves(demanda_semana, OUT_DIR)

    print("[13] Generando heatmap semanal...")
    plot_weekly_heatmap(demanda_semana, OUT_DIR)


    # --------------------------------------------------
    # 5. MAPAS
    # --------------------------------------------------
    print("\n[14] Generando mapa global...")
    map_global(demanda, DATA_DIR, OUT_DIR)

    print("[15] Generando mapa por zona...")
    map_local(demanda, DATA_DIR, OUT_DIR)

    # --------------------------------------------------
    # 6. CLUSTERING
    # --------------------------------------------------
    print("\n[16] Ejecutando clustering...")
    clustering_analysis(demanda, DATA_DIR, OUT_DIR)

    print("\n===================================")
    print("PIPELINE COMPLETADO")
    print("Resultados en:", OUT_DIR)
    print("===================================")

    spark.stop()


if __name__ == "__main__":
    main()