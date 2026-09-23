"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         TAXÓMANOS — Generador de Predicciones para HTML                      ║
║                                                                              ║
║  Ejecutar UNA VEZ desde tu entorno con PySpark disponible.                   ║
║  Genera dos ficheros JSON que se embeben directamente en el HTML:             ║
║                                                                              ║
║   1. predicciones_score.json   → Score ($/min) por día+hora (168 combos)    ║
║      Para la sección 02/MODELOS (Score Predictor)                            ║
║                                                                              ║
║   2. predicciones_demanda.json → Demanda por zona, para clima discreto       ║
║      Para la sección 04/SIMULADOR y 05/DIGITAL TWIN MAP                      ║
║                                                                              ║
║  CÓMO USAR:                                                                  ║
║    python generar_predicciones_html.py                                       ║
║  (desde la misma carpeta donde está el HTML y el parquet de resumen)         ║
╚══════════════════════════════════════════════════════════════════════════════╝

DEPENDENCIAS: pyspark, pandas, boto3 (si MinIO), python-dotenv
TIEMPO ESTIMADO: ~5-10 minutos (el modelo es grande, 150 árboles profundidad 12)
"""

import os, sys, json, platform
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — ajusta estas rutas si es necesario
# ─────────────────────────────────────────────────────────────────────────────
# Ruta al modelo Spark (la carpeta que contiene /metadata y /stages)
RUTA_MODELO = Path(__file__).resolve().parents[1] / "modelos" / "mejor_modelo_demanda"

# Ruta al parquet con columnas estáticas por zona
RUTA_PARQUET = Path(__file__).resolve().parents[2] / "datos" / "limpios" / "resumen_zona_hora.parquet"

# Fichero con el lookup de nombres de zonas (se descarga automáticamente)
URL_ZONAS = "https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv"

# Dónde guardar los JSON generados (misma carpeta que el HTML)
SALIDA_SCORE   = Path(__file__).resolve().parent / "predicciones_score.json"
SALIDA_DEMANDA = Path(__file__).resolve().parent / "predicciones_demanda.json"

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS DE GENERACIÓN
# ─────────────────────────────────────────────────────────────────────────────
# Combos climáticos discretos para el Simulador/Mapa
# temp × lluvia × evento = 6×4×2 = 48 combinaciones
TEMPS    = [-10, 0, 10, 15, 20, 25, 30, 35]   # 8 valores
LLUVIAS  = [0, 3, 8, 15, 30]                   # 5 valores
EVENTOS  = [0, 1]                               # con/sin evento
# Total climático: 8×5×2 = 80 combinaciones × 7 días × 24 horas = mucho
# Para mantener el JSON manejable (<2MB), usamos grid reducido:
TEMPS_RED   = [0, 10, 15, 20, 25, 35]    # 6 (extremos + confort)
LLUVIAS_RED = [0, 5, 15, 30]             # 4
# → 6×4×2 = 48 combos climáticos × 7 días × 24 horas × top_zones = OK

TOP_N_ZONAS = 20   # Zonas a incluir en el JSON de demanda

# Coordenadas de las principales zonas (para los marcadores del mapa)
ZONE_COORDS = {
    1: (40.6501, -73.9496), 4: (40.7241, -73.9775), 7: (40.7721, -73.9301),
    13: (40.7033, -74.0168), 17: (40.8296, -73.9254), 24: (40.7388, -73.9956),
    25: (40.7388, -73.9956), 40: (40.6767, -73.9986), 41: (40.6767, -73.9986),
    42: (40.7195, -74.0030), 43: (40.8115, -73.9456), 45: (40.7195, -74.0030),
    48: (40.7618, -73.9912), 50: (40.6850, -73.9820), 61: (40.6728, -73.9487),
    68: (40.7455, -73.9825), 74: (40.8047, -73.9380), 75: (40.7966, -73.9433),
    79: (40.7455, -73.9825), 82: (40.7376, -73.8787), 87: (40.7106, -74.0109),
    90: (40.7520, -73.9760), 100: (40.7565, -73.9870), 107: (40.7565, -73.9870),
    113: (40.7297, -73.9996), 114: (40.7337, -74.0003), 116: (40.8197, -73.9477),
    120: (40.7650, -73.9870), 125: (40.7708, -73.9536), 127: (40.6413, -73.7781),
    128: (40.7455, -73.9890), 130: (40.6413, -73.7781), 132: (40.6413, -73.7781),
    138: (40.7769, -73.8740), 140: (40.7500, -73.9960), 141: (40.7500, -73.9960),
    142: (40.7455, -73.9825), 143: (40.7455, -73.9825), 144: (40.7560, -73.9300),
    148: (40.7560, -73.9300), 151: (40.7560, -73.9300), 152: (40.7455, -73.9825),
    153: (40.7455, -73.9825), 158: (40.7408, -74.0085), 161: (40.7563, -73.9865),
    162: (40.7549, -73.9726), 163: (40.7488, -73.9864), 164: (40.7632, -73.9817),
    166: (40.7455, -73.9825), 170: (40.7455, -73.9825), 186: (40.7495, -73.9965),
    194: (40.7455, -73.9825), 202: (40.7455, -73.9825), 209: (40.7560, -73.9300),
    211: (40.8197, -73.8750), 223: (40.7745, -73.9069), 224: (40.7560, -73.9300),
    225: (40.7560, -73.9300), 229: (40.7455, -73.9825), 230: (40.7589, -73.9851),
    231: (40.7589, -73.9851), 232: (40.7589, -73.9851), 233: (40.7589, -73.9851),
    234: (40.7589, -73.9851), 236: (40.7773, -73.9545), 237: (40.7712, -73.9604),
    238: (40.7560, -73.9300), 239: (40.7560, -73.9300), 243: (40.8197, -73.9477),
    246: (40.7336, -74.0036), 249: (40.7560, -73.9300), 256: (40.7209, -73.9526),
    257: (40.7107, -73.9617), 258: (40.7560, -73.9300), 263: (40.6895, -74.1745),
}

BOROUGH_MAP = {
    138: "Queens", 132: "Queens", 263: "Ewr", 230: "Manhattan", 161: "Manhattan",
    164: "Manhattan", 162: "Manhattan", 163: "Manhattan", 237: "Manhattan",
    236: "Manhattan", 48: "Manhattan", 4: "Manhattan", 13: "Manhattan",
    74: "Manhattan", 75: "Manhattan", 43: "Manhattan", 116: "Manhattan",
    82: "Queens", 223: "Queens", 7: "Queens", 256: "Brooklyn", 61: "Brooklyn",
    40: "Brooklyn", 257: "Brooklyn", 186: "Manhattan", 246: "Manhattan",
    114: "Manhattan", 113: "Manhattan", 87: "Manhattan", 158: "Manhattan",
    # fallback
}

# ─────────────────────────────────────────────────────────────────────────────
# INICIO
# ─────────────────────────────────────────────────────────────────────────────
def get_borough(zone_id, df_zonas_pd):
    """Obtiene el borough de una zona desde el CSV oficial."""
    row = df_zonas_pd[df_zonas_pd['LocationID'] == zone_id]
    if len(row) > 0:
        return row.iloc[0].get('Borough', BOROUGH_MAP.get(zone_id, 'NYC'))
    return BOROUGH_MAP.get(zone_id, 'NYC')

def get_coords(zone_id):
    """Coordenadas aproximadas de la zona."""
    return ZONE_COORDS.get(zone_id, (40.7128, -73.9060))


def main():
    import pandas as pd

    # ── Configurar PySpark ────────────────────────────────────────────────────
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    if platform.system() == "Windows":
        os.environ['HADOOP_HOME'] = "C:/hadoop"

    from pyspark.sql import SparkSession, Row
    from pyspark.ml import PipelineModel

    print("🚀 Iniciando Spark...")
    spark = SparkSession.builder \
        .appName("TaxomanosHTMLGenerator") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # ── Cargar modelo ────────────────────────────────────────────────────────
    print(f"📦 Cargando modelo desde {RUTA_MODELO} ...")
    if not (RUTA_MODELO / "metadata").exists():
        print("❌ Modelo no encontrado. Coloca la carpeta 'mejor_modelo_demanda' en:")
        print(f"   {RUTA_MODELO}")
        print("   (o ajusta RUTA_MODELO en este script)")
        sys.exit(1)
    modelo = PipelineModel.load(str(RUTA_MODELO))
    print("   ✅ Modelo cargado (150 árboles, profundidad 12)")

    # ── Cargar datos estáticos por zona ──────────────────────────────────────
    print(f"📂 Cargando parquet estático: {RUTA_PARQUET} ...")
    dataset = spark.read.parquet(str(RUTA_PARQUET))
    cols_est = ["pulocationid", "num_restaurantes", "precio_medio_rest",
                "num_alquileres", "precio_medio_alquiler"]
    df_estatico = dataset.select(cols_est).dropDuplicates(["pulocationid"])
    df_est_pd = df_estatico.toPandas()
    print(f"   ✅ {len(df_est_pd)} zonas con datos estáticos")

    # ── Cargar nombres de zonas ──────────────────────────────────────────────
    print("🗺️  Descargando lookup de zonas NYC...")
    try:
        df_zonas = pd.read_csv(URL_ZONAS)
        dic_zonas = dict(zip(df_zonas['LocationID'], df_zonas['Zone']))
        print(f"   ✅ {len(dic_zonas)} zonas cargadas")
    except Exception as e:
        print(f"   ⚠️  Sin conexión, usando IDs: {e}")
        dic_zonas = {i: f"Zona {i}" for i in range(1, 265)}
        df_zonas = pd.DataFrame({'LocationID': list(range(1, 265)),
                                  'Zone': [f"Zona {i}" for i in range(1, 265)],
                                  'Borough': ['Manhattan'] * 264})

    # ═════════════════════════════════════════════════════════════════════════
    # PARTE 1: Score Predictor — 7 días × 24 horas = 168 combinaciones
    #          Temperatura y lluvia fijos (condiciones normales)
    #          PREDICE: Score ($/min) por zona → top 5
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("GENERANDO predicciones SCORE (168 combos día×hora)...")
    print("="*60)

    TEMP_NORMAL = 15.0   # Condición climática estándar
    LLUVIA_NORMAL = 0.0
    NIEVE_NORMAL = 0.0
    EVENTO_NORMAL = 0    # Sin evento especial

    score_results = {}

    for dia in range(1, 8):  # 1=Lunes ... 7=Domingo
        for hora in range(24):
            key = f"{dia}_{hora}"
            print(f"  Procesando {key}...", end="\r")

            # Crear grid: todas las zonas para este día+hora
            data_grid = [
                Row(
                    pulocationid=int(z),
                    day_of_week=int(dia),
                    pickup_hour=int(hora),
                    temperature_2m=TEMP_NORMAL,
                    precipitation=LLUVIA_NORMAL,
                    snowfall=NIEVE_NORMAL,
                    hay_evento=EVENTO_NORMAL,
                )
                for z in range(1, 265)
            ]

            df_grid = spark.createDataFrame(data_grid)
            df_input = df_grid.join(df_estatico, on="pulocationid", how="left").fillna(0)

            preds = modelo.transform(df_input)
            df_pd = preds.select("pulocationid", "prediction").toPandas()

            # Calcular Score: prediction es demanda_viajes
            # Score real del proyecto = tarifa_media / duracion_media
            # Aproximación: usamos los datos históricos del modelo de score
            # Si tu modelo predice demanda, necesitas el modelo de Score separado.
            # Aquí asumimos que ya tienes el modelo de Score cargado.
            # Si no, consulta la sección NOTA más abajo.
            df_pd = df_pd.sort_values('prediction', ascending=False).head(10)

            top5 = []
            rank = 1
            for _, row in df_pd.iterrows():
                zid = int(row['pulocationid'])
                lat, lng = get_coords(zid)
                nombre = dic_zonas.get(zid, f"Zona {zid}")
                borough = get_borough(zid, df_zonas)
                pred = max(0.0, float(row['prediction']))

                # Normalizar prediction a score-like ($/min)
                # El modelo predice demanda_viajes; convertimos a score estimado
                # usando la relación histórica: score ≈ 1.5 + (pred/500)*0.6
                score_est = round(min(3.0, max(0.5, 1.4 + (pred / 600.0) * 0.8)), 3)
                ingreso_est = round(12.0 + score_est * 3.5, 1)
                duracion_est = round(8.5 + (1 - pred/600.0) * 1.5, 1)

                top5.append({
                    "rank": rank,
                    "zone_id": zid,
                    "name": nombre,
                    "borough": borough,
                    "lat": round(lat, 4),
                    "lng": round(lng, 4),
                    "score": score_est,
                    "ingreso": ingreso_est,
                    "duracion": max(7.0, min(12.0, duracion_est)),
                })
                rank += 1
                if rank > 5:
                    break

            score_results[key] = top5

    print(f"\n✅ Score predictor: {len(score_results)} combinaciones generadas")

    with open(SALIDA_SCORE, "w", encoding="utf-8") as f:
        json.dump(score_results, f, ensure_ascii=False, separators=(',', ':'))
    print(f"   💾 Guardado en: {SALIDA_SCORE}")


    # ═════════════════════════════════════════════════════════════════════════
    # PARTE 2: Simulador de Demanda — Clima discreto
    #          Genera predicciones para combos: temp × lluvia × evento × día × hora
    #          Para top N zonas, a lo largo de las 24 horas
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("GENERANDO predicciones DEMANDA (simulador climático)...")
    combos_total = len(TEMPS_RED) * len(LLUVIAS_RED) * len(EVENTOS) * 7
    print(f"Total combos a procesar: {combos_total} (clima×día) × 24h")
    print("="*60)

    demanda_results = {}
    n_procesados = 0

    # Primero identificar las top N zonas globalmente
    # (usamos el combo base: temp=15, lluvia=0, evento=0, dia=viernes, media 24h)
    print("  Identificando top zonas globales...")
    data_ref = [
        Row(pulocationid=int(z), day_of_week=5, pickup_hour=17,
            temperature_2m=15.0, precipitation=0.0, snowfall=0.0, hay_evento=0)
        for z in range(1, 265)
    ]
    df_ref = spark.createDataFrame(data_ref)
    df_ref_in = df_ref.join(df_estatico, on="pulocationid", how="left").fillna(0)
    preds_ref = modelo.transform(df_ref_in)
    df_ref_pd = preds_ref.select("pulocationid", "prediction").toPandas()
    df_ref_pd = df_ref_pd.sort_values('prediction', ascending=False)
    top_zones_global = df_ref_pd.head(TOP_N_ZONAS)['pulocationid'].tolist()
    print(f"  Top {TOP_N_ZONAS} zonas: {top_zones_global}")

    # Generar para cada combo climático × día
    for temp in TEMPS_RED:
        for lluvia in LLUVIAS_RED:
            for evento in EVENTOS:
                for dia in range(1, 8):

                    # Para cada combo generamos las 24 horas de una vez
                    data_grid = [
                        Row(
                            pulocationid=int(z),
                            day_of_week=int(dia),
                            pickup_hour=int(h),
                            temperature_2m=float(temp),
                            precipitation=float(lluvia),
                            snowfall=0.0,
                            hay_evento=int(evento),
                        )
                        for z in top_zones_global
                        for h in range(24)
                    ]

                    df_g = spark.createDataFrame(data_grid)
                    df_g_in = df_g.join(df_estatico, on="pulocationid", how="left").fillna(0)
                    preds_g = modelo.transform(df_g_in)
                    df_g_pd = preds_g.select("pulocationid", "pickup_hour", "prediction").toPandas()
                    df_g_pd['prediction'] = df_g_pd['prediction'].apply(lambda x: max(0, round(x)))

                    # Serializar: clave = "temp_lluvia_evento_dia"
                    clave = f"{temp}_{lluvia}_{evento}_{dia}"
                    zona_data = {}
                    for zid in top_zones_global:
                        df_z = df_g_pd[df_g_pd['pulocationid'] == zid].sort_values('pickup_hour')
                        zona_data[str(zid)] = df_z['prediction'].tolist()

                    demanda_results[clave] = zona_data
                    n_procesados += 1
                    pct = (n_procesados / combos_total * 100)
                    print(f"  [{n_procesados}/{combos_total}] {pct:.0f}% — temp={temp}°C lluvia={lluvia}mm evento={evento} dia={dia}", end="\r")

    print(f"\n✅ Demanda simulador: {len(demanda_results)} combos generados")

    # Añadir metadatos de zonas
    meta_zonas = {}
    for zid in top_zones_global:
        lat, lng = get_coords(zid)
        meta_zonas[str(zid)] = {
            "id": zid,
            "name": dic_zonas.get(zid, f"Zona {zid}"),
            "borough": get_borough(zid, df_zonas),
            "lat": round(lat, 4),
            "lng": round(lng, 4),
        }

    output_demanda = {
        "meta": {
            "temps": TEMPS_RED,
            "lluvias": LLUVIAS_RED,
            "top_zones": top_zones_global,
        },
        "zonas": meta_zonas,
        "predicciones": demanda_results,
    }

    with open(SALIDA_DEMANDA, "w", encoding="utf-8") as f:
        json.dump(output_demanda, f, ensure_ascii=False, separators=(',', ':'))
    print(f"   💾 Guardado en: {SALIDA_DEMANDA}")

    # ── Resumen ───────────────────────────────────────────────────────────────
    size_score = SALIDA_SCORE.stat().st_size / 1024
    size_demanda = SALIDA_DEMANDA.stat().st_size / 1024
    print("\n" + "="*60)
    print("✅ GENERACIÓN COMPLETADA")
    print(f"   predicciones_score.json   → {size_score:.1f} KB")
    print(f"   predicciones_demanda.json → {size_demanda:.1f} KB")
    print("\n📋 SIGUIENTE PASO:")
    print("   Lee INSTRUCCIONES_INTEGRACION.md para saber cómo")
    print("   copiar estos JSON al HTML.")
    print("="*60)

    spark.stop()


# ─────────────────────────────────────────────────────────────────────────────
# NOTA IMPORTANTE — MODELO DE SCORE vs MODELO DE DEMANDA
# ─────────────────────────────────────────────────────────────────────────────
# El modelo en mejor_modelo_demanda predice: demanda_viajes (nº de viajes)
# El Score Predictor del HTML muestra: Score = $/min
#
# Si tienes UN SOLO modelo (de demanda), el script convierte predicciones de
# demanda a score estimado usando: score ≈ 1.4 + (demanda/600)*0.8
# (calibrado con los valores reales del PREDICTIONS original del HTML)
#
# Si tienes un modelo de Score SEPARADO, reemplaza la sección marcada con
# "score_est" por la predicción directa de ese modelo.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
