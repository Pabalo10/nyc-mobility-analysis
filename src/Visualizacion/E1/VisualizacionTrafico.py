import pandas as pd
import plotly.express as px
import folium
from folium.plugins import HeatMap
import os
from pathlib import Path

# 
# CONFIGURACIÓN DE RUTAS RELATIVAS
# Obtiene la ruta del directorio donde está ESTE script (src/Visualizacion)
DIRECTORIO_ACTUAL = Path(__file__).resolve().parent

# PROJECT_ROOT sube dos niveles para llegar a la raíz (Entrega1_Pd2)
PROJECT_ROOT = DIRECTORIO_ACTUAL.parents[2]

# Define el archivo de entrada (Ahora es PARQUET, en la carpeta datos/limpios)
ARCHIVO_ENTRADA = PROJECT_ROOT / "datos" / "limpios" / "dataset_trafico_vis_ready.parquet"

# Define la carpeta de salida (se creará dentro del directorio del script)
CARPETA_SALIDA = DIRECTORIO_ACTUAL / "Reporte_Trafico_NYC"

# Crea la carpeta si no existe
if not CARPETA_SALIDA.exists():
    print(f"Creando carpeta de salida: {CARPETA_SALIDA}")
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)


def cargar_datos(ruta):
    print(f"Cargando dataset desde:\n{ruta}")
    # Cargamos solo columnas necesarias para optimizar memoria
    cols = ['latitude', 'longitude', 'Vol', 'hora_entera', 'Boro', 'street', 'SegmentID']
    try:
        # Cambiado a read_parquet ya que ahora nuestro archivo limpio es parquet
        df = pd.read_parquet(ruta, columns=cols)
        print(f"Datos cargados: {len(df)} registros.")
        return df
    except Exception as e:
        print(f"Error cargando datos: {e}")
        print("Asegúrate de haber ejecutado antes el script de transformación que genera el Parquet.")
        return None


def generar_mapa_animado(df):
    """
    Crea un mapa interactivo (Plotly) que muestra la evolución del tráfico por hora.
    Agrupa los datos para mostrar un 'Día Promedio'.
    """
    print("Generando: Mapa Animado de Tráfico (Plotly)")

    # Agregamos datos: Promedio de volumen por Segmento y Hora
    df_agg = df.groupby(['SegmentID', 'hora_entera', 'street', 'Boro', 'latitude', 'longitude'])[
        'Vol'].mean().reset_index()

    # Redondeamos volumen para que se vea limpio
    df_agg['Vol'] = df_agg['Vol'].round(0)

    # Ordenamos por hora para que la animación fluya bien
    df_agg = df_agg.sort_values('hora_entera')

    # Configurar el Mapa
    fig = px.scatter_mapbox(
        df_agg,
        lat="latitude",
        lon="longitude",
        color="Vol",  # El color depende del volumen
        size="Vol",  # El tamaño del círculo también
        hover_name="street",  # Al pasar el ratón muestra la calle
        hover_data={"Boro": True, "Vol": True, "latitude": False, "longitude": False, "SegmentID": False},
        animation_frame="hora_entera",  # Barra de tiempo
        color_continuous_scale=px.colors.sequential.Plasma,  # Colores tipo fuego/neón
        size_max=15,  # Tamaño máximo de los puntos
        zoom=10,
        center={"lat": 40.73, "lon": -73.93},  # Centro de NYC
        mapbox_style="carto-positron",  # Estilo de mapa limpio
        title="Evolución del Tráfico Promedio por Hora en NYC"
    )

    archivo = CARPETA_SALIDA / "1_mapa_animado_trafico.html"
    fig.write_html(archivo)
    print(f"Mapa animado guardado en:\n{archivo}")
    print("\n")


def generar_mapa_calor(df):
    """
    Crea un mapa estático de calor (Folium) para identificar 'Hotspots'.
    """
    print("Generando: Mapa de Calor General (Folium)")

    # Filtramos: Nos interesan los puntos donde REALMENTE hay tráfico
    umbral = df['Vol'].quantile(0.50)  # Solo el 50% superior de tráfico
    df_calor = df[df['Vol'] > umbral].copy()

    # Agrupamos por ubicación para obtener la intensidad promedio total
    data_heatmap = df_calor.groupby(['latitude', 'longitude'])['Vol'].mean().reset_index()

    # Lista de listas [lat, lon, peso] requerida por Folium
    heat_data = data_heatmap[['latitude', 'longitude', 'Vol']].values.tolist()

    # Creo mapa base
    m = folium.Map(location=[40.73, -73.93], zoom_start=11, tiles="CartoDB positron")

    # Añado capa de calor
    HeatMap(heat_data, radius=10, blur=15, max_zoom=13).add_to(m)

    archivo = CARPETA_SALIDA / "2_mapa_calor_zonas.html"
    m.save(str(archivo)) # Folium requiere string para la ruta
    print(f"Mapa de calor guardado en:\n{archivo}")
    print("\n")


def generar_grafico_lineas(df):
    """
    Gráfico de líneas comparativo: Hora vs Volumen por Distrito (Boro).
    """
    print("Generando: Comparativa de Distritos (Plotly)")

    # Agrupar por Hora y Distrito
    df_b = df.groupby(['hora_entera', 'Boro'])['Vol'].mean().reset_index()

    fig = px.line(
        df_b,
        x='hora_entera',
        y='Vol',
        color='Boro',
        markers=True,
        title="Perfil Diario de Tráfico por Distrito (Volumen Promedio)",
        labels={'hora_entera': 'Hora del Día (0-23)', 'Vol': 'Volumen Promedio de Vehículos', 'Boro': 'Distrito'}
    )

    fig.update_layout(xaxis=dict(tickmode='linear', dtick=1))  # Mostrar todas las horas en eje X

    archivo = CARPETA_SALIDA / "3_grafico_horarios_distritos.html"
    fig.write_html(archivo)
    print(f"Gráfico de líneas guardado en:\n{archivo}")
    print("\n")


def main():
    df = cargar_datos(ARCHIVO_ENTRADA)

    if df is not None:
        generar_mapa_animado(df)
        generar_mapa_calor(df)
        generar_grafico_lineas(df)

        print("\nPROCESO FINALIZADO")
        print(f"Tus reportes están listos en la carpeta:\n{CARPETA_SALIDA}")
        print("Abre los archivos HTML con tu navegador para explorarlos.")


if __name__ == "__main__":
    main()