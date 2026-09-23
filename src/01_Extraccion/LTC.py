import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path

"""
En este Script cargamos los principales datos de viajes en Taxi en la ciudad de Nueva York en el año 2023.
Hemos parseado los datos por la primera semana de cada mes, debido al volumen total de datos que había disponibles
y haremos el estudio posterior teniendo esto en cuenta.
"""

# ===============================
# Rutas
# ===============================

BASE_DIR = Path(__file__).resolve()
PROJECT_ROOT = BASE_DIR.parents[2]   # ajusta si tu estructura cambia
DATA_DIR = PROJECT_ROOT / "datos" / "crudos"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / "sampled_nyc_yellow_taxi_2023.csv"

# ===============================
# Configuración de nuestra API
# ===============================

URL = "https://data.cityofnewyork.us/resource/4b4i-vvec.json"

"""
COLUMAS que usaremos:

    - vendorid -> empresa que registro el viaje (1: Creative Mobile Technologies, 2: VeriFone)
    - tpep_pickup_datetime -> fecha y hora de inicio del viaje
    - tpep_dropoff_datetime -> fecha y hora de final del viaje
    - passenger_count -> número de pasajeros que recoge el taxi
    - trip_distance -> distancia recorrida en MILLAS
    - pulocationid -> id de la zona de recogida (cruzar con Taxi Zone Lookup Table)
    - dolocationid -> id de la zona de destino
    - payment_type -> tipo de método de pago 
    - fare_amount -> tarifa base del taxi
    - extra -> recargos adicionales por horario nocturno, hora punta, etc.
    - tip_amount -> cantidad de propina dada por el pasajero
    - tolls_amount -> numero de peajes durante el viaje
    - congestion_surcharge -> recargo por congestión (tráfico)
    - total_amount -> importe total pagado por el pasajero

"""

COLUMNS_NEEDED = [
    "vendorid",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "pulocationid",
    "dolocationid",
    "payment_type",
    "fare_amount",
    "extra",
    "tip_amount",
    "tolls_amount",
    "congestion_surcharge",
    "total_amount"
]

# Para la consulta API, necesitamos TODAS las columnas en orden:
COLUMNS_API = [
    "vendorid",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "ratecodeid",        #no nos interesa
    "store_and_fwd_flag", #no nos interesa
    "pulocationid",
    "dolocationid",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",           #no nos interesa
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge", #no nos interesa
    "total_amount",
    "congestion_surcharge",
    "airport_fee"        #no nos interesa
]

LIMIT = 50000          # tamaño de bloque
DAYS_PER_MONTH = 14     # muestreo: primeros N días del mes
SLEEP_TIME = 0.2       # respeto al API

# ===============================
# Descarga Datos Taxi Amarillo 2023
# ===============================

def download_yellow_taxi_sample_2023():
    first_write = True

    print("📅 Descargando Yellow Taxi 2023 (muestreo mensual por días)")

    for month in range(1, 13):
        month_start = datetime(2023, month, 1)
        print(f"\n🟦 Mes {month:02d} | muestreo {DAYS_PER_MONTH} días")

        for day in range(DAYS_PER_MONTH):
            start = month_start + timedelta(days=day)
            end = start + timedelta(days=1)

            offset = 0
            total_day = 0

            print(f"  📆 Día {start.date()}")

            while True:
                params = {
                    "$limit": LIMIT,
                    "$offset": offset,
                    "$select": ",".join(COLUMNS_API),
                    "$where": (
                        f"tpep_pickup_datetime >= '{start.isoformat()}' "
                        f"AND tpep_pickup_datetime < '{end.isoformat()}'"
                    )
                }

                try:
                    r = requests.get(URL, params=params, timeout=30)
                    r.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print(f"    ⚠️ Error API: {e}")
                    break

                data = r.json()

                if not data:
                    break

                df = pd.DataFrame(data)
                df = df[COLUMNS_NEEDED]

                df.to_csv(
                    OUTPUT_FILE,
                    mode="a",
                    header=first_write,
                    index=False
                )

                first_write = False
                rows = len(df)
                total_day += rows
                offset += LIMIT

                print(f"     → {total_day} filas acumuladas")

                time.sleep(SLEEP_TIME)

    print(f"\n✅ Dataset Yellow Taxi 2023 muestreado guardado en:\n{OUTPUT_FILE}")


# ===============================
# ▶️ Main
# ===============================
if __name__ == "__main__":
    download_yellow_taxi_sample_2023()
