import pandas as pd
from pathlib import Path

# Buscamos el archivo en la misma ruta que usamos antes
BASE_DIR = Path(__file__).resolve()
PROJECT_ROOT = BASE_DIR.parents[2]
PARQUET_PATH = PROJECT_ROOT / "datos" / "limpios" / "resumen_zona_hora.parquet"

def comprobar_datos():
    print(f"Buscando archivo en: {PARQUET_PATH}")
    
    if not PARQUET_PATH.exists():
        print(" ❌ ERROR: El archivo no existe. Revisa la ruta o vuelve a ejecutar agregaciones.py")
        return

    # Cargamos el dataframe
    df = pd.read_parquet(PARQUET_PATH)
    
    print("\n" + "="*60)
    print(" 🌟 ARCHIVO MAESTRO CARGADO CON ÉXITO 🌟")
    print("="*60)
    
    # 1. Tamaño del dataset
    print(f"\n📏 TAMAÑO DEL DATASET:")
    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")
    
    # 2. Tipos de datos y nulos (Es crucial que no haya nulos y los tipos sean int/float)
    print("\n🔍 TIPOS DE DATOS Y VALORES NULOS:")
    info_df = pd.DataFrame({
        'Tipo': df.dtypes,
        'Nulos': df.isna().sum()
    })
    print(info_df)
    
    # 3. Muestra de las primeras 5 filas (para ver las columnas nuevas)
    print("\n📄 PRIMERAS 5 FILAS:")
    print(df.head().to_string())
    
    # 4. Comprobación de rangos (Sanity Check)
    print("\n🔬 CONTROL DE RANGOS (Sanity Check):")
    print("\n--- VARIABLES BASE (TAXIS) ---")
    print(f"- Horas (pickup_hour): Min {df['pickup_hour'].min()} | Max {df['pickup_hour'].max()} (Debería ser 0 a 23)")
    print(f"- Días (day_of_week): Min {df['day_of_week'].min()} | Max {df['day_of_week'].max()} (Debería ser 1 a 7)")
    print(f"- Demanda mínima: {df['demanda_viajes'].min()} viajes | máxima: {df['demanda_viajes'].max()} viajes")
    print(f"- Total de zonas únicas: {df['pulocationid'].nunique()} (Deberían ser unas 260+)")

    print("\n--- VARIABLES TEMPORALES (CLIMA Y EVENTOS) ---")
    if 'temperature_2m' in df.columns:
        print(f"- Temperatura (ºC): Min {df['temperature_2m'].min():.1f} | Max {df['temperature_2m'].max():.1f} | Media {df['temperature_2m'].mean():.1f}")
        print(f"- Lluvia máxima registrada en una hora: {df['precipitation'].max():.1f} mm")
        print(f"- Nieve máxima registrada en una hora: {df['snowfall'].max():.1f} mm")
        print(f"- Horas con Eventos (1=Sí, 0=No): \n{df['hay_evento'].value_counts().to_string()}")

    print("\n--- VARIABLES ESPACIALES (GEO) ---")
    if 'num_restaurantes' in df.columns:
        zonas_con_rest = (df.groupby('pulocationid')['num_restaurantes'].max() > 0).sum()
        zonas_con_alq = (df.groupby('pulocationid')['num_alquileres'].max() > 0).sum()
        print(f"- Zonas que contienen restaurantes: {zonas_con_rest} zonas")
        print(f"- Zonas que contienen pisos de alquiler: {zonas_con_alq} zonas")
        print(f"- Máx restaurantes en UNA sola zona: {df['num_restaurantes'].max()}")
        print(f"- Máx alquileres (pisos) en UNA sola zona: {df['num_alquileres'].max()}")
        print(f"- Rango Precio Medio Restaurantes: {df['precio_medio_rest'].min():.2f} a {df['precio_medio_rest'].max():.2f} (Categoría)")
        print(f"- Rango Precio Medio Alquileres: {df['precio_medio_alquiler'].min():.2f}$ a {df['precio_medio_alquiler'].max():.2f}$")

if __name__ == "__main__":
    comprobar_datos()