import pandas as pd
from pathlib import Path


# Rutas
BASE_DIR = Path(__file__).resolve()
PROJECT_ROOT = BASE_DIR.parents[3]

DATA_DIR = PROJECT_ROOT / "datos" / "limpios"

FHV_PATH = DATA_DIR / "fhv_2023_clean.parquet"
YLC_PATH = DATA_DIR / "nyc_taxi_clean.parquet"

ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv"

def auditar_datos():
    print("1. Cargando datos")
    df_fhv = pd.read_parquet(FHV_PATH)
    df_ylc = pd.read_parquet(YLC_PATH)
    df_lookup = pd.read_csv(ZONE_LOOKUP_URL)

    # Unir un poco de datos para probar
    sample_fhv = df_fhv[['pulocationid']].head(1000).copy()
    sample_ylc = df_ylc[['pulocationid']].head(1000).copy()
    
    print(f"\n2. Tipos de datos detectados")
    print(f"FHV ID tipo: {sample_fhv['pulocationid'].dtype}")
    print(f"Taxi ID tipo: {sample_ylc['pulocationid'].dtype}")
    print(f"Lookup ID tipo: {df_lookup['LocationID'].dtype}")

    # Intentar el merge 
    df_combined = pd.concat([sample_fhv, sample_ylc])
    
    # MERGE
    merged = df_combined.merge(df_lookup, left_on='pulocationid', right_on='LocationID', how='left')
    
    print("\n3. Resultados del cruce (Borough)")
    print(merged['Borough'].value_counts(dropna=False))
    
    nulos = merged['Borough'].isna().sum()
    total = len(merged)
    print(f"\n FILAS PERDIDAS (NaN): {nulos} de {total} ({nulos/total*100:.1f}%)")

if __name__ == "__main__":
    auditar_datos()