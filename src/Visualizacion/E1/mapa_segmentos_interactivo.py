import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import TimestampedGeoJson
from pathlib import Path
import json


BASE_DIR = Path(__file__).resolve()
PROJECT_ROOT = BASE_DIR.parents[3]

RESUMEN_HORA_PATH = PROJECT_ROOT / "datos" / "limpios" / "resumen_zona_hora.parquet"
ZONES_PATH = PROJECT_ROOT / "datos" / "limpios" / "taxi_zones.shp"

OUTPUT_DIR = PROJECT_ROOT / "datos" / "salidas_html"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def cargar_datos():
    resumen = pd.read_parquet(RESUMEN_HORA_PATH)
    zones = gpd.read_file(ZONES_PATH)

    if "LocationID" not in zones.columns:
        candidates = [c for c in zones.columns if c.lower() == "locationid"]
        if candidates:
            zones = zones.rename(columns={candidates[0]: "LocationID"})
        else:
            raise ValueError("No se encontró LocationID")

    zones = zones.to_crs(epsg=4326)

    return resumen, zones


def crear_features(resumen, zones, metric):

    features = []

    for hora in range(24):

        data_hora = resumen[resumen["pickup_hour"] == hora]
        gdf = zones.merge(data_hora, on="LocationID", how="left").fillna(0)

        for _, row in gdf.iterrows():

            valor = row[metric]

            feature = {
                "type": "Feature",
                "geometry": json.loads(row["geometry"].to_json()),
                "properties": {
                    "time": f"2023-01-01T{hora:02d}:00:00",
                    "style": {
                        "color": "black",
                        "weight": 0.3,
                        "fillColor": "#08306b",
                        "fillOpacity": min(0.8, valor / (gdf[metric].max() + 1))
                    },
                    "popup": f"""
                    <b>Zona:</b> {row.get('zone', '')}<br>
                    <b>Hora:</b> {hora}<br>
                    <b>FHV:</b> {row.get('FHV', 0)}<br>
                    <b>YLC:</b> {row.get('YLC', 0)}<br>
                    <b>Ratio:</b> {row.get('ratio', 0):.2f}
                    """
                }
            }

            features.append(feature)

    return features


def crear_mapa(metric):

    resumen, zones = cargar_datos()

    m = folium.Map(
        location=[40.7128, -74.0060],
        zoom_start=11,
        tiles="cartodbpositron"
    )

    features = crear_features(resumen, zones, metric)

    TimestampedGeoJson(
        {
            "type": "FeatureCollection",
            "features": features,
        },
        period="PT1H",
        add_last_point=False,
        auto_play=False,
        loop=False,
        max_speed=1,
        loop_button=True,
        date_options="HH",
        time_slider_drag_update=True,
    ).add_to(m)

    output_path = OUTPUT_DIR / f"mapa_slider_{metric}.html"
    m.save(str(output_path))

    print("Mapa generado:", output_path)


if __name__ == "__main__":

    # Elegir métrica aquí:
    # "FHV"
    # "YLC"
    "ratio"

    crear_mapa("ratio")
