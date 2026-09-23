"""
Cleaning_FHV.py
----------------
Limpieza del dataset High Volume FHV Trip Data (NYC 2023)
Procesamiento y guardado en parquet
Proyecto de Datos II
"""

import pandas as pd
import pyarrow.dataset as ds
from pathlib import Path

print("=== Cleaning_FHV.py EJECUTADO ===")
print("Archivo:", __file__)

# =====================
# Paths
# =====================
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_RAW = BASE_DIR / "datos" / "crudos"
DATA_PROCESSED = BASE_DIR / "datos" / "limpios"

INPUT_FILE = DATA_RAW / "fhv_2023_sampled_full.parquet"
OUTPUT_DIR = DATA_PROCESSED / "fhv_2023_clean_parquet"
OUTPUT_FILE_FINAL = DATA_PROCESSED / "fhv_2023_clean.parquet"

# =====================
# Cleaning
# =====================
NUMERIC_COLS = [
    "pulocationid",
    "dolocationid",
    "trip_miles",
    "base_passenger_fare",
    "tolls",
    "tips",
    "driver_pay",
]

def clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fechas
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"], errors="coerce")

    # Duración
    df["trip_duration_min"] = (
        df["dropoff_datetime"] - df["pickup_datetime"]
    ).dt.total_seconds() / 60

    df = df[df["trip_duration_min"] > 0]

    # Conversión numérica
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filtros básicos
    df = df[df["trip_miles"] >= 0]
    df = df[df["base_passenger_fare"] >= 0]
    df = df[df["driver_pay"] >= 0]

    return df

def main():
    print(">>> Entré a main() <<<")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chunk_size = 1_000_000
    total_rows = 0

    print("Iniciando limpieza FHV por chunks (PARQUET)...")

    dataset = ds.dataset(INPUT_FILE, format="parquet")

    for i, batch in enumerate(dataset.to_batches(batch_size=chunk_size)):
        print(f"Procesando chunk {i}")

        clean = clean_chunk(batch.to_pandas())
        total_rows += len(clean)

        output_file = OUTPUT_DIR / f"part_{i:03d}.parquet"
        clean.to_parquet(output_file, index=False)

    print("Limpieza finalizada")
    print(f"Filas finales: {total_rows}")
    print(f"Archivos generados en: {OUTPUT_DIR}")

    # Unificación parquet
    print("Leyendo parquet limpio por partes para unificar...")
    parts = sorted(OUTPUT_DIR.glob("part_*.parquet"))
    df_final = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)

    print("Guardando parquet unificado...")
    df_final.to_parquet(OUTPUT_FILE_FINAL, index=False)

    print("Dataset final:", OUTPUT_FILE_FINAL)
    print("Filas:", len(df_final))

if __name__ == "__main__":
    main()
