import numpy as np
import pandas as pd
from pathlib import Path
import time

"""
    Este script lo usaremos para centralizar y preparar los datos de movilidad de taxis tradicionales (YLC)
    y de vehículos de transporte con conductor (FHV/Uber) de Nueva York, agregándolos por hora y combinándolos 
    con información meteorológica. 
    
    Se crea un dataset limpio y unificado que contiene, para cada hora:
    - 'YLC'
    - 'FHV' : el número total de viajes por tipo de servicio
    - 'total' : numero total de viajes
    - 'market_share' : el market share de FHV
    - 'ratio' : el ratio FHV/YLC
    - 'temperature_2m' : temperatura
    - 'precipitation' : precipitación
    - 'rain' : precipitación
    - 'snowfall' : nieve
    - 'snow_depth' : profundidad de nieve
    
"""


# =====================================
# 📂 RUTAS
# =====================================

BASE_DIR = Path(__file__).resolve()
PROJECT_ROOT = BASE_DIR.parents[2]

DATA_DIR = PROJECT_ROOT / "datos" / "limpios"

FHV_PATH = DATA_DIR / "fhv_2023_clean.parquet"
LTC_PATH = DATA_DIR / "nyc_taxi_clean.parquet"
WEATHER_PATH = DATA_DIR / "nyc_weather_2023_nyc_weather_2023_first_half.csv"

OUTPUT_PATH = DATA_DIR / "hourly_aggregate.parquet"

# ==========================================
# 1. CARGA DE DATOS
# ==========================================

COLUMNS_LTC = [
    'tpep_pickup_datetime',
    'trip_distance',
    'fare_amount',
    'total_amount',
    'trip_duration_min'
]

COLUMNS_FHV = [
    'pickup_datetime',
    'trip_miles',
    'base_passenger_fare',
    'tolls',
    'tips',
    'trip_duration_min'
]


def load_data():

    init_time = time.time()

    ltc = pd.read_parquet(LTC_PATH, columns = COLUMNS_LTC)
    fhv = pd.read_parquet(FHV_PATH, columns = COLUMNS_FHV)
    weather = pd.read_csv(WEATHER_PATH)

    end_time = time.time()
    print(f"Tiempo de carga: {(end_time-init_time):.4f} \n")

    return ltc, fhv, weather


# ==========================================
# 2. PREPARACIÓN TEMPORAL
# ==========================================

def prepare_data(ltc, fhv):

    init_time = time.time()

    #Normalizamos nombres de las columnas para que no haya conflictos
    ltc.rename(columns={
        'tpep_pickup_datetime': 'pickup_datetime'
    }, inplace=True)

    fhv.rename(columns={
        'trip_miles': 'trip_distance',
        'base_passenger_fare': 'fare_amount'
    }, inplace=True)

    #añadimos columnas faltantes que puedan ser de valor
    fhv['total_amount'] = fhv['fare_amount'] + fhv['tolls']

    #Aseguramos formato datetime en todas las columnas de tipo fecha
    ltc["pickup_datetime"] = pd.to_datetime(ltc["pickup_datetime"])
    fhv["pickup_datetime"] = pd.to_datetime(fhv["pickup_datetime"])
    #weather.index = pd.to_datetime(weather.index)

    ltc["datetime_hour"] = ltc["pickup_datetime"].dt.floor("h")
    fhv["datetime_hour"] = fhv["pickup_datetime"].dt.floor("h")

    end_time = time.time()
    print(f"Tiempo de preparado: {(end_time - init_time):.4f} \n")

    return ltc, fhv


# ==========================================
# 3. AGREGACIÓN POR SERVICIO
# ==========================================

def aggregate_service(ltc, fhv):

    init_time = time.time()

    ltc_agg = (
        ltc.groupby(["datetime_hour"])
        .size()
        .reset_index(name="YLC")
    )

    fhv_agg = (
        fhv.groupby(["datetime_hour"])
        .size()
        .reset_index(name="FHV")
    )

    # Merge ambos
    merged = pd.merge(
        ltc_agg,
        fhv_agg,
        on=["datetime_hour"],
        how="outer"
    ).fillna(0)
    merged["YLC"] = merged["YLC"].astype(int)
    merged["FHV"] = merged["FHV"].astype(int)

    merged["total"] = merged["YLC"] + merged["FHV"]

    merged["market_share"] = np.where(
        merged["total"] > 0,
        merged["FHV"] / merged["total"],
        0
    )

    merged["ratio"] = merged["FHV"] / (merged["YLC"] + 1)

    end_time = time.time()
    print(f"Tiempo de merge de servicios: {(end_time - init_time):.4f} \n")

    return merged


# ==========================================
# 4. PREPARAR WEATHER
# ==========================================

def prepare_weather(weather):

    init_time = time.time()

    weather["datetime_hour"] = pd.to_datetime(weather["date"])

    # Quitar timezone
    weather["datetime_hour"] = weather["datetime_hour"].dt.tz_localize(None)

    weather = weather.drop(columns=["date"])

    # Floor por seguridad
    weather["datetime_hour"] = weather["datetime_hour"].dt.floor("h")

    end_time = time.time()
    print(f"Tiempo de preparacion de weather: {(end_time - init_time):.4f} \n")

    return weather


# ==========================================
# 5. MERGE CON CLIMA
# ==========================================

def merge_weather(mobility, weather):

    init_time = time.time()

    merged = mobility.merge(
        weather,
        on="datetime_hour",
        how="left"
    )

    columnas = ['temperature_2m', 'precipitation', 'rain', 'snowfall', 'snow_depth']
    merged[columnas] = merged[columnas].fillna(0)

    merged = merged.sort_values("datetime_hour")
    merged.set_index("datetime_hour", inplace=True)
    merged = merged.drop_duplicates()

    end_time = time.time()
    print(f"Tiempo de merge de weather: {(end_time - init_time):.4f} \n")

    return merged


# ==========================================
# 6. GUARDAR DATASET FINAL
# ==========================================

def save_dataset(df):

    df.to_parquet(OUTPUT_PATH)#, index=False
    print("✅ Dataset agregado guardado correctamente.")


# ==========================================
# MAIN
# ==========================================

def main():

    init_time = time.time()

    print("📦 Cargando datos...")
    ltc, fhv, weather = load_data()

    print("⚙️ Preparando movilidad...")
    ltc, fhv = prepare_data(ltc, fhv)

    print("📊 Agregando movilidad...")
    mobility = aggregate_service(ltc, fhv)

    print("🌦 Preparando clima...")
    weather = prepare_weather(weather)

    print("🔗 Merge movilidad + clima...")
    final_df = merge_weather(mobility, weather)

    print("💾 Guardando...")
    save_dataset(final_df)

    print("🚀 Listo.")

    end_time = time.time()
    print(f"Tiempo del proceso entero: {(end_time - init_time):.4f} \n")


if __name__ == "__main__":
    main()
