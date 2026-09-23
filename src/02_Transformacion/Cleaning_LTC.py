import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from pathlib import Path

# ===============================
#  Rutas del proyecto
# ===============================

BASE_DIR = Path(__file__).resolve()
PROJECT_ROOT = BASE_DIR.parents[2]  # ajusta si tu estructura cambia

# Ruta de CARGA
RAW_DATA_PATH = PROJECT_ROOT / "datos" / "crudos" / "sampled_nyc_yellow_taxi_2023.csv"

# Ruta de DESTINO
CLEAN_DATA_DIR = PROJECT_ROOT / "datos" / "limpios"
CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = CLEAN_DATA_DIR / "nyc_taxi_clean.parquet"


# ===============================
# 🔎 Exploración de los datos
# ===============================
def explore_data(df):
    """Explorar los datos descargados"""
    if df is None or df.empty:
        print("No hay datos para explorar")
        return

    print("\n" + "=" * 50)
    print("🔎 EXPLORACIÓN DE DATOS")
    print("=" * 50)

    print("\n📏 Nº de filas del dataframe")
    nrows, ncols = df.shape
    print(nrows)

    print("\n📄 Primeras 5 filas:")
    print(df.head())

    print("\n📊 Tipos de datos:")
    print(df.dtypes)

    print("\n🧮 Estadísticas (columnas numéricas):")
    print(df.describe())

    print("\n⚠️ Valores nulos por columna:")
    print(df.isnull().sum())

    print("\n🔢 Valores distintos por columna:")
    print(df[["vendorid", "pulocationid", "dolocationid", "payment_type"]].nunique())


# ===============================
# ❓ Consultas e info básicas
# ===============================
def basic_queries(df):
    print("\n" + "=" * 50)
    print("📈 CONSULTAS ÚTILES")
    print("=" * 50)

    if "trip_distance" in df.columns:
        df["trip_distance"] = pd.to_numeric(df["trip_distance"], errors="coerce")
        print("\n📍 Top 5 viajes más largos:")
        print(
            df.nlargest(5, "trip_distance")[
                ["trip_distance", "fare_amount", "total_amount"]
            ]
        )

    if "vendorid" in df.columns:
        print("\n🏢 Viajes por vendor:")
        print(df["vendorid"].value_counts())

    if "payment_type" in df.columns:
        print("\n💳 Distribución de tipos de pago:")
        print(df["payment_type"].value_counts())


# ===============================
# 🧹 Limpieza de datos
# ===============================
def clean_taxi_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y prepara los datos de NYC taxi.

    Parámetros:
        - df: pd.DataFrame
            Dataframe en crudo de los datos extraidos de los viajes en
            taxi en la ciudad de Nueva York en el año 2023. Contendrá
            las siguientes columnas:

            - 'vendorid': ID de la empresa que registra el viaje (numérico - entero)
            - 'tpep_pickup_datetime': Fecha y hora de recogida (string o datetime)
            - 'tpep_dropoff_datetime': Fecha y hora de entrega (string o datetime)
            - 'passenger_count': Número de pasajeros (numérico - float)
            - 'trip_distance': Distancia del viaje en millas (numérico - float)
            - 'pulocationid': ID de la zona de recogida (numérico - entero)
            - 'dolocationid': ID de la zona de final del viaje (numérico - entero)
            - 'payment_type': Tipo de método pago (numérico - entero)
            - 'fare_amount': Tarifa base del taxi (numérico - float)
            - 'extra': Recargos adicionales realizados (numérico - float)
            - 'tip_amount': Cantidad de propina dada por el pasajero (numérico - float)
            - 'tolls_amount': Número de peajes durante el viaje (numérico - float)
            - 'congestion_surcharge': Recargo por congestión (numérico - float)
            - 'total_amount': Precio total cobrado en USD (numérico - float)

    Devuelve:
        df: pd.DataFrame
            DataFrame limpio y listo para utilizar

    """

    print("🧹 Iniciando limpieza de datos ")

    # ---------------------
    # Conversion de Fechas
    # ---------------------
    date_cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce", format="%Y-%m-%dT%H:%M:%S.%f")

    # ---------------------------
    # Eliminacion de valores inválidos y nulos
    # ---------------------------
    df = df.dropna(subset=date_cols)
    df = df[df['trip_distance'] > 0]
    df = df[df['total_amount'] > 0]
    df = df[df['passenger_count'] > 0]
    df = df[df['fare_amount'] > 0]

    # ---------------------------
    # Creacion de variables derivadas
    # ---------------------------
    df['trip_duration_min'] = ((df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()/60).round(2)
    df = df[df['trip_duration_min'] > 0]

    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df["pickup_weekday"] = df["tpep_pickup_datetime"].dt.dayofweek

    df["revenue_per_mile"] = (df["total_amount"] / df["trip_distance"]).round(2)

    # ---------------------------
    # Finalizacion
    # ---------------------------
    df = df.reset_index(drop=True)
    return df


def main():
    print("=" * 40)
    print("🚕 LIMPIEZA Y EXPLORACIÓN - NYC TAXI")
    print("=" * 40)

    if not RAW_DATA_PATH.exists():
        print(f"❌ No se ha encontrado el archivo: {RAW_DATA_PATH}")
        return

    print(f"\n📂 Cargando datos desde {RAW_DATA_PATH}")

    chunksize = 500000
    chunks = pd.read_csv(RAW_DATA_PATH, sep=',', chunksize=chunksize)

    df_list = []
    for i, chunk in enumerate(chunks):
        print(f"Procesando chunk {i}")

        chunk_clean = clean_taxi_data(chunk)
        df_list.append(chunk_clean)
    
    df = pd.concat(df_list, ignore_index = True)

    explore_data(df)
    basic_queries(df)

    print(f"💾 Guardando datos limpios en:\n{OUTPUT_PATH}")
    #df_clean.to_csv(OUTPUT_PATH, index=False)
    df.to_parquet(OUTPUT_PATH, index=False)


    print(f"📊 Número de filas finales: {len(df)}")



if __name__ == "__main__":
    main()
