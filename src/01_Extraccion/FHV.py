import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path

# ===============================
# Rutas del proyecto
# ===============================

BASE_DIR = Path(__file__).resolve()
PROJECT_ROOT = BASE_DIR.parents[2]

RAW_DIR = PROJECT_ROOT / "datos" / "crudos"
DAILY_DIR = RAW_DIR / "nyc_fhv_2023_sampled"
FINAL_FILE = RAW_DIR / "fhv_2023_sampled_full.parquet"

DAILY_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# Configuración API Socrata
# ===============================

URL = "https://data.cityofnewyork.us/resource/u253-aew4.json"

COLUMNS = [
    "pickup_datetime", #fecha y hora de recogida
    "dropoff_datetime", #fecha y hora de la dejada
    "pulocationid", #id de la zona de recogida
    "dolocationid", #id de la zona de dejada
    "trip_miles", #numero de MILLAS que se recorren en el viaje
    "base_passenger_fare", #tarifa base del viaje
    "tolls", #importe total de los peajes durante el viaje
    "tips", #propina al conductor
    "driver_pay" #pago al conductor
]

LIMIT = 50000          # tamaño de chunk
DAYS_PER_MONTH = 14     # muestreo: primeros 14 días
SLEEP_TIME = 0.2       # para no matar el API


# ===============================
# Función de tipado correcto
# ===============================

def clean_types(df):
    """
    Convierte tipos correctamente para análisis posterior.
    Fundamental en datasets NYC.
    """

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"], errors="coerce")

    numeric_cols = [
        "pulocationid",
        "dolocationid",
        "trip_miles",
        "base_passenger_fare",
        "tolls",
        "tips",
        "driver_pay"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ===============================
# Descarga FHV 2023
# ===============================

def download_fhv_sample_2023():

    print("Descargando FHV 2023 (14 días por mes)")

    for month in range(1, 13):

        month_start = datetime(2023, month, 1)
        print(f"\nMes {month:02d}")

        for day in range(DAYS_PER_MONTH):

            start = month_start + timedelta(days=day)
            end = start + timedelta(days=1)

            offset = 0
            total_day = 0
            daily_frames = []

            print(f"  Día {start.date()}")

            while True:

                params = {
                    "$limit": LIMIT,
                    "$offset": offset,
                    "$select": ",".join(COLUMNS),
                    "$where": (
                        f"pickup_datetime >= '{start.isoformat()}' "
                        f"AND pickup_datetime < '{end.isoformat()}'"
                    )
                }

                try:
                    r = requests.get(URL, params=params, timeout=30)
                    r.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print(f"    Error API: {e}")
                    break

                data = r.json()

                if not data:
                    break

                df = pd.DataFrame(data)
                df = clean_types(df)

                daily_frames.append(df)

                rows = len(df)
                total_day += rows
                offset += LIMIT

                print(f"    → {total_day} filas acumuladas")

                time.sleep(SLEEP_TIME)

            # ===============================
            # Guardado diario en parquet
            # ===============================

            if daily_frames:
                daily_df = pd.concat(daily_frames, ignore_index=True)

                file_name = f"fhv_2023_{start.strftime('%m_%d')}.parquet"
                output_path = DAILY_DIR / file_name

                daily_df.to_parquet(output_path, index=False, compression="snappy")

                print(f"    Guardado: {file_name} ({len(daily_df)} filas)")

    print("\nDescarga diaria finalizada.")


# ===============================
# Unión final en un solo parquet
# ===============================

def merge_all_parquets():

    print("\nUniendo todos los parquet en uno...")

    parquet_files = sorted(DAILY_DIR.glob("*.parquet"))

    if not parquet_files:
        print("No se encontraron archivos para unir.")
        return

    dataframes = []

    for file in parquet_files:
        print(f"Leyendo {file.name}")
        df = pd.read_parquet(file)
        dataframes.append(df)

    full_df = pd.concat(dataframes, ignore_index=True)

    full_df.to_parquet(FINAL_FILE, index=False, compression="snappy")

    print(f"\nArchivo final creado:")
    print(FINAL_FILE)
    print(f"Total filas: {len(full_df)}")


# ===============================
# Main
# ===============================

if __name__ == "__main__":

    download_fhv_sample_2023()
    merge_all_parquets()