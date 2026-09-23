import pandas as pd
import geopandas as gpd
import folium
from pathlib import Path
import json
import time
import numpy as np
from branca.element import Element

BASE_DIR = Path(__file__).resolve()
PROJECT_ROOT = BASE_DIR.parents[3]

RESUMEN_HORA_PATH = PROJECT_ROOT / "datos" / "limpios" / "resumen_zona_hora.parquet"
ZONES_PATH = PROJECT_ROOT / "datos" / "crudos" / "taxi_zones.shp"

OUTPUT_DIR = PROJECT_ROOT / "datos" / "salidas_html"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ================================
# SEGMENTOS
# ================================
def asignar_segmento(hora: int) -> str:
    if 5 <= hora < 8:
        return "Primera mañana"
    elif 8 <= hora < 11:
        return "Hora pico mañana"
    elif 11 <= hora < 16:
        return "Mediodía"
    elif 16 <= hora < 21:
        return "Regreso a casa"
    else:
        return "Noche / Madrugada"


# ================================
# PREPARAR DATOS
# ================================
def preparar_datos():
    print(" Cargando parquet...")
    t0 = time.time()
    resumen = pd.read_parquet(RESUMEN_HORA_PATH)
    print(f"    Parquet cargado en {round(time.time()-t0,2)}s")

    # Tipos para matchear con el GeoJSON
    resumen["LocationID"] = resumen["LocationID"].astype(int)
    resumen["pickup_hour"] = resumen["pickup_hour"].astype(int)

    print(" Cargando shapefile...")
    t0 = time.time()
    zones = gpd.read_file(ZONES_PATH)

    if "LocationID" not in zones.columns:
        raise RuntimeError(
            f" El shapefile no tiene columna 'LocationID'. Tiene: {list(zones.columns)}"
        )

    zones["LocationID"] = zones["LocationID"].astype(int)

    # CRS a WGS84 para Leaflet
    zones = zones.to_crs(epsg=4326)

    # Simplificar geometría para que el HTML no sea gigante
    zones["geometry"] = zones["geometry"].simplify(0.001, preserve_topology=True)

    # Reducir columnas del GeoJSON (más liviano)
    keep_cols = ["LocationID", "geometry"]
    for c in ["zone", "Zone", "borough", "Borough", "service_zone"]:
        if c in zones.columns and c not in keep_cols:
            keep_cols.append(c)
    zones = zones[keep_cols]

    print(f"   ✔ Shapefile listo en {round(time.time()-t0,2)}s")

    print("🔹 Asignando segmentos...")
    resumen["segmento"] = resumen["pickup_hour"].apply(asignar_segmento)

    print("🔹 Agregando por segmento...")
    t0 = time.time()
    agg = (
        resumen
        .groupby(["segmento", "LocationID"])[["FHV", "YLC"]]
        .sum()
        .reset_index()
    )
    print(f"   ✔ Agregado en {round(time.time()-t0,2)}s")

    print("🔹 Calculando market share...")
    denom = (agg["FHV"] + agg["YLC"]).to_numpy()
    agg["market_share"] = np.where(denom > 0, agg["FHV"] / denom, 0.0).round(4)

    # Para escalar colores en cada segmento
    max_by_seg = (
        agg.groupby("segmento")[["FHV", "YLC"]]
        .max()
        .fillna(0)
        .astype(float)
        .to_dict(orient="index")
    )
    for seg in max_by_seg:
        max_by_seg[seg]["market_share"] = 1.0

    print("🔹 Convirtiendo a diccionario liviano...")
    t0 = time.time()
    data_dict = {}
    for segmento in agg["segmento"].unique():
        df_seg = agg[agg["segmento"] == segmento][
            ["LocationID", "FHV", "YLC", "market_share"]
        ].copy()

        # CLAVE: en JS el lookup va como string
        df_seg["LocationID"] = df_seg["LocationID"].astype(str)
        data_dict[segmento] = df_seg.set_index("LocationID").to_dict(orient="index")

    print(f"   ✔ Diccionario creado en {round(time.time()-t0,2)}s")

    return zones, data_dict, max_by_seg


# ================================
# MAPA
# ================================
def crear_mapa():
    print("🚀 Iniciando generación de mapa...")
    zones, segment_data, max_by_seg = preparar_datos()

    print("🔹 Generando geojson base...")
    t0 = time.time()
    geojson_str = zones.to_json()  # string JSON listo para inyectar en JS
    print(f"   ✔ GeoJSON base listo en {round(time.time()-t0,2)}s")

    segment_json = json.dumps(segment_data)
    max_json = json.dumps(max_by_seg)

    print("🔹 Creando mapa folium...")
    m = folium.Map(
        location=[40.7128, -74.0060],
        zoom_start=11,
        tiles="cartodbpositron"
    )

    map_id = m.get_name()

    # Panel HTML (solo HTML)
    panel_html = """
    <div style="position: fixed; top:10px; left:50px; z-index:9999;
                background:white; padding:12px; border-radius:8px;
                box-shadow:0 2px 6px rgba(0,0,0,0.3); font-family:Arial;">

        <b>Segmento horario:</b><br>
        <select id="segmentSelect">
            <option>Primera mañana</option>
            <option selected>Hora pico mañana</option>
            <option>Mediodía</option>
            <option>Regreso a casa</option>
            <option>Noche / Madrugada</option>
        </select>

        <br><br>

        <b>Métrica:</b><br>
        <input type="radio" name="metric" value="FHV" checked> 🔵 FHV<br>
        <input type="radio" name="metric" value="YLC"> 🟡 YLC<br>
        <input type="radio" name="metric" value="market_share"> 🔵🟡 Market Share
    </div>
    """
    m.get_root().html.add_child(Element(panel_html))

    js = f"""
(function() {{
    window.addEventListener('load', function() {{

        var map = {map_id};
        var geoData = {geojson_str};
        var segmentData = {segment_json};
        var maxBySeg = {max_json};

        function clamp(x, a, b) {{
            return Math.max(a, Math.min(b, x));
        }}

        function getColor(value, metric, maxV) {{
            if (!maxV || maxV <= 0) maxV = 1;

            if (metric === "FHV") {{
                var t = clamp(value / maxV, 0, 1);
                var alpha = (value === 0) ? 0.05 : (0.15 + 0.85 * t);
                return "rgba(0,0,255," + alpha.toFixed(3) + ")";
            }}

            if (metric === "YLC") {{
                var t = clamp(value / maxV, 0, 1);
                var alpha = (value === 0) ? 0.05 : (0.15 + 0.85 * t);
                return "rgba(255,204,0," + alpha.toFixed(3) + ")";
            }}

            // market_share 0..1 (YLC domina -> amarillo, empate -> blanco, FHV domina -> azul)
            if (metric === "market_share") {{
                var v = clamp(value, 0, 1);

                // d=0 en 0.5 (empate), d=1 en extremos (0 o 1)
                var d = Math.abs(v - 0.5) / 0.5;

                if (v < 0.5) {{
                    // blanco -> amarillo
                    var b = Math.floor(255 * (1 - d));   // 255 en empate, 0 en dominancia fuerte
                    return "rgb(255,255," + b + ")";
                }} else {{
                    // blanco -> azul
                    var rg = Math.floor(255 * (1 - d));  // 255 en empate, 0 en dominancia fuerte
                    return "rgb(" + rg + "," + rg + ",255)";
                }}
            }}

            return "#ffffff";
        }}

        function fmtValue(v, metric) {{
            if (metric === "market_share") return (100*v).toFixed(1) + "%";
            return (v || 0).toLocaleString();
        }}

        function pickProp(props, keys) {{
            for (var i=0; i<keys.length; i++) {{
                if (props[keys[i]] !== undefined && props[keys[i]] !== null) return props[keys[i]];
            }}
            return null;
        }}

        function tooltipHtml(feature, segmento, metric, segObj) {{
            var props = feature.properties || {{}};
            var loc = String(props.LocationID);

            var zoneName = pickProp(props, ["zone", "Zone", "zone_name", "name"]) || ("LocationID " + loc);
            var borough  = pickProp(props, ["borough", "Borough"]) || "";

            var d = (segObj && segObj[loc]) ? segObj[loc] : null;

            var fhv = d ? (d["FHV"] || 0) : 0;
            var ylc = d ? (d["YLC"] || 0) : 0;
            var ms  = d ? (d["market_share"] || 0) : 0;

            var headline = "<b>" + zoneName + "</b>" + (borough ? ("<br><span style='color:#555'>" + borough + "</span>") : "");
            var segLine = "<br><span style='color:#333'>Segmento:</span> " + segmento;

            var currentVal = d ? (d[metric] || 0) : 0;

            var vals =
                "<hr style='margin:6px 0'>" +
                "<div style='font-size:12px'>" +
                "FHV: <b>" + fmtValue(fhv, "FHV") + "</b><br>" +
                "YLC: <b>" + fmtValue(ylc, "YLC") + "</b><br>" +
                "Market share: <b>" + fmtValue(ms, "market_share") + "</b><br>" +
                "<span style='color:#666'>Métrica actual:</span> <b>" + metric + "</b> = <b>" +
                    fmtValue(currentVal, metric) + "</b>" +
                "</div>";

            return "<div style='min-width:180px'>" + headline + segLine + vals + "</div>";
        }}

        // Capa GeoJSON con tooltip + hover highlight (SIN resetStyle)
        var layer = L.geoJson(geoData, {{
            style: function(feature) {{
                return {{
                    weight: 0.3,
                    color: "black",
                    fillOpacity: 0.85,
                    fillColor: "#ffffff"
                }};
            }},
            onEachFeature: function(feature, lyr) {{
                lyr.bindTooltip("Cargando...", {{
                    sticky: true,
                    direction: "auto",
                    opacity: 0.95
                }});

                lyr.on("mouseover", function(e) {{
                    e.target.setStyle({{
                        weight: 2,
                        color: "#111"
                    }});
                    if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) {{
                        e.target.bringToFront();
                    }}
                }});

                lyr.on("mouseout", function(e) {{
                    // NO resetStyle porque borra el fillColor dinámico
                    e.target.setStyle({{
                        weight: 0.3,
                        color: "black"
                    }});
                }});
            }}
        }}).addTo(map);

        function updateMap() {{
            var segmento = document.getElementById("segmentSelect").value;
            var metric = document.querySelector('input[name="metric"]:checked').value;

            var segObj = segmentData[segmento] || {{}};
            var maxV = (maxBySeg[segmento] && maxBySeg[segmento][metric]) ? maxBySeg[segmento][metric] : 1;

            layer.eachLayer(function(l) {{
                var loc = String(l.feature.properties.LocationID);
                var d = segObj[loc];

                var v = d ? (d[metric] || 0) : 0;

                l.setStyle({{
                    fillColor: getColor(v, metric, maxV)
                }});

                if (l.getTooltip && l.getTooltip()) {{
                    l.setTooltipContent(tooltipHtml(l.feature, segmento, metric, segObj));
                }}
            }});
        }}

        document.getElementById("segmentSelect").addEventListener("change", updateMap);
        document.querySelectorAll('input[name="metric"]').forEach(function(r) {{
            r.addEventListener("change", updateMap);
        }});

        updateMap();
    }});
}})();
"""



    m.get_root().script.add_child(Element(js))

    output_path = OUTPUT_DIR / "mapa_segmentos_interactivo.html"

    print(" Guardando HTML...")
    t0 = time.time()
    m.save(str(output_path))
    print(f"    HTML guardado en {round(time.time()-t0,2)}s")

    print(" MAPA GENERADO CORRECTAMENTE:", output_path)


if __name__ == "__main__":
    crear_mapa()
