import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import os

# CONFIGURACIÓN DE RUTAS RELATIVAS
# BASE_DIR apunta a la carpeta donde está este script (src/Visualizacion)
BASE_DIR = Path(__file__).resolve().parent

# PROJECT_ROOT sube dos niveles para llegar a la raíz (Entrega1_Pd2)
PROJECT_ROOT = BASE_DIR.parents[2]

# Archivos de Demanda (Taxis/Uber) apuntando a datos/limpios
FHV_PATH = PROJECT_ROOT / "datos" / "limpios" / "fhv_2023_clean.parquet"
YLC_PATH = PROJECT_ROOT / "datos" / "limpios" / "nyc_taxi_clean.parquet"

# Archivo de Tráfico apuntando a datos/limpios
TRAFICO_PATH = PROJECT_ROOT / "datos" / "limpios" / "dataset_trafico_vis_ready.parquet" 

# Lookup de zonas (URL Oficial)
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv"


# CARGA Y PREPROCESAMIENTO DE DEMANDA
def cargar_demanda():
    print("Cargando datos de Taxis y Ubers...")
    try:
        df_fhv = pd.read_parquet(FHV_PATH)
        df_ylc = pd.read_parquet(YLC_PATH)
    except FileNotFoundError:
        print("No se encontraron los archivos de Taxis/Uber en las rutas especificadas.")
        print(f"Buscando en:\n{FHV_PATH}\n{YLC_PATH}")
        return None

    # Estandarización
    df_fhv = df_fhv.rename(columns={"trip_miles": "trip_distance", "pickup_datetime": "datetime"})
    df_ylc = df_ylc.rename(columns={"tpep_pickup_datetime": "datetime", "tpep_dropoff_datetime": "dropoff_datetime"})
    
    # Asegurar datetime
    df_fhv["datetime"] = pd.to_datetime(df_fhv["datetime"])
    df_ylc["datetime"] = pd.to_datetime(df_ylc["datetime"])

    df_fhv["tipo_servicio"] = "Uber/FHV"
    df_ylc["tipo_servicio"] = "Taxi Amarillo"

    # Concatenar solo columnas necesarias
    cols = ["datetime", "pulocationid", "tipo_servicio"]
    df_total = pd.concat([df_fhv[cols], df_ylc[cols]], ignore_index=True)
    
    # Extraer hora
    df_total["hora_entera"] = df_total["datetime"].dt.hour
    
    return df_total


# CARGA DE TRÁFICO
def cargar_trafico():
    print("Cargando datos de Tráfico...")
    try:
        df = pd.read_parquet(TRAFICO_PATH)
        return df
    except FileNotFoundError:
        print(f"No se encontró el archivo de tráfico en:\n{TRAFICO_PATH}")
        return None

# UNIFICACIÓN ESPACIAL (Boroughs)
def mapear_zonas_a_borough(df_demanda):
    print("Mapeando zonas de taxi a Boroughs...")
    try:
        df_lookup = pd.read_csv(ZONE_LOOKUP_URL)
        df_lookup = df_lookup[['LocationID', 'Borough']]
        
        # Merge con la demanda
        df_merged = df_demanda.merge(df_lookup, left_on='pulocationid', right_on='LocationID', how='left')
        return df_merged
    except Exception as e:
        print(f"No se pudo mapear zonas: {e}")
        return df_demanda


# VISUALIZACIONES
def plot_ritmo_ciudad(df_traf, df_dem):
    """
    Gráfico de Doble Eje: Tráfico vs Demanda por Hora
    """
    traf_agg = df_traf.groupby("hora_entera")["Vol"].mean().reset_index()
    dem_agg = df_dem.groupby(["hora_entera", "tipo_servicio"]).size().reset_index(name="viajes")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Línea de Tráfico
    fig.add_trace(
        go.Scatter(x=traf_agg["hora_entera"], y=traf_agg["Vol"], name="Volumen Tráfico (Promedio)",
                   fill='tozeroy', line=dict(color='gray', width=1, dash='dot'), opacity=0.3),
        secondary_y=False
    )

    # Líneas de Demanda
    colors = {"Uber/FHV": "#00CC96", "Taxi Amarillo": "#F6D55C"}
    for servicio in dem_agg["tipo_servicio"].unique():
        data = dem_agg[dem_agg["tipo_servicio"] == servicio]
        fig.add_trace(
            go.Scatter(x=data["hora_entera"], y=data["viajes"], name=f"Viajes {servicio}",
                       line=dict(color=colors.get(servicio, "blue"), width=3)),
            secondary_y=True
        )

    fig.update_layout(
        title_text="<b>El Ritmo de NYC:</b> Tráfico vs Demanda por Hora",
        xaxis_title="Hora del Día (0-23)",
        template="plotly_white"
    )
    
    fig.update_yaxes(title_text="Volumen de Tráfico", secondary_y=False)
    fig.update_yaxes(title_text="Cantidad de Viajes", secondary_y=True)

    return fig

def plot_comparativa_boroughs_detallado(df_traf, df_dem):
    """
    Gráfico 2: Bar Chart Comparativo NORMALIZADO (Porcentaje)
    SEPARANDO Taxis vs Ubers vs Tráfico (3 barras por distrito)
    """
    if "Borough" not in df_dem.columns:
        print("La columna 'Borough' no existe en el dataset de demanda.")
        return None

    # TRÁFICO
    traf_boro = df_traf.groupby("Boro")["Vol"].sum().reset_index()
    total_trafico = traf_boro["Vol"].sum()
    traf_boro["Porcentaje"] = (traf_boro["Vol"] / total_trafico) * 100
    traf_boro["Tipo"] = "Tráfico"
    traf_boro = traf_boro.rename(columns={"Boro": "Borough"}) 

    # DEMANDA (Separada por Tipo)
    dem_boro = df_dem.groupby(["Borough", "tipo_servicio"]).size().reset_index(name="Viajes")
    totales_por_servicio = df_dem.groupby("tipo_servicio").size().to_dict()
    
    dem_boro["Porcentaje"] = dem_boro.apply(
        lambda x: (x["Viajes"] / totales_por_servicio[x["tipo_servicio"]]) * 100, 
        axis=1
    )
    dem_boro = dem_boro.rename(columns={"tipo_servicio": "Tipo"})
    
    # UNIR Y GRAFICAR
    df_comp = pd.concat([
        traf_boro[["Borough", "Porcentaje", "Tipo"]], 
        dem_boro[["Borough", "Porcentaje", "Tipo"]]
    ], ignore_index=True)
    
    df_comp = df_comp[df_comp["Borough"] != "Unknown"]
    
    colores = {
        "Tráfico": "gray", 
        "Taxi Amarillo": "#F6D55C", 
        "Uber/FHV": "#00CC96"
    }

    fig = px.bar(df_comp, x="Borough", y="Porcentaje", color="Tipo", barmode="group",
                 title="<b>Disparidad Espacial Detallada:</b> Taxis vs Ubers vs Tráfico",
                 color_discrete_map=colores,
                 text_auto='.1f') 
    
    fig.update_layout(
        template="plotly_white", 
        yaxis_title="Porcentaje del Total de su categoría (%)",
        legend_title_text="Categoría"
    )
    return fig

def plot_comparativa_boroughs_agregado(df_traf, df_dem):
    """
    Gráfico 3: Bar Chart Comparativo NORMALIZADO (Porcentaje)
    AGRUPANDO toda la demanda (Taxis + Ubers) vs Tráfico (2 barras por distrito)
    """
    if "Borough" not in df_dem.columns:
        print("La columna 'Borough' no existe en el dataset de demanda.")
        return None
    
    # Agregación Tráfico
    traf_boro = df_traf.groupby("Boro")["Vol"].sum().reset_index()
    total_trafico = traf_boro["Vol"].sum()
    traf_boro["Porcentaje"] = (traf_boro["Vol"] / total_trafico) * 100 
    traf_boro["Tipo"] = "Tráfico (% del total)"
    traf_boro = traf_boro.rename(columns={"Boro": "Borough"}) 

    # Agregación Demanda (TOTAL)
    dem_boro = df_dem.groupby("Borough")["tipo_servicio"].count().reset_index()
    total_demanda = dem_boro["tipo_servicio"].sum()
    dem_boro["Porcentaje"] = (dem_boro["tipo_servicio"] / total_demanda) * 100 
    dem_boro["Tipo"] = "Demanda Total (% del total)"

    # Unir
    df_comp = pd.concat([
        traf_boro[["Borough", "Porcentaje", "Tipo"]], 
        dem_boro[["Borough", "Porcentaje", "Tipo"]]
    ], ignore_index=True)
    
    df_comp = df_comp[df_comp["Borough"] != "Unknown"]

    # Graficar
    fig = px.bar(df_comp, x="Borough", y="Porcentaje", color="Tipo", barmode="group",
                 title="<b>Disparidad Espacial General:</b> Tráfico vs Demanda Total",
                 color_discrete_map={"Tráfico (% del total)": "gray", "Demanda Total (% del total)": "orange"},
                 text_auto='.1f')
    
    fig.update_layout(template="plotly_white", yaxis_title="Porcentaje del Total (%)")
    return fig


# MAIN
def main():
    # 1. Cargar Datos
    df_demand = cargar_demanda()
    df_traffic = cargar_trafico()

    if df_demand is not None and df_traffic is not None:
        
        # 2. Mapear Zonas
        df_demand = mapear_zonas_a_borough(df_demand)

        # 3. Preparar carpeta de salida (Se crea si no existe)
        output_dir = BASE_DIR / "Reporte_Trafico_NYC"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nDirectorio de reportes: {output_dir}")

        print("Generando gráficos interactivos...")
        
        # Gráfico 1: Series Temporales
        fig1 = plot_ritmo_ciudad(df_traffic, df_demand)
        file_path_1 = output_dir / "trafico_vs_demanda_horario.html"
        fig1.write_html(file_path_1)
        print(f"Gráfico 1 guardado en: {file_path_1}")
        # fig1.show() 

        # Gráfico 2: Comparativa Detallada (3 barras)
        fig2 = plot_comparativa_boroughs_detallado(df_traffic, df_demand)
        if fig2:
            file_path_2 = output_dir / "comparativa_boroughs_detallado.html"
            fig2.write_html(file_path_2)
            print(f"Gráfico 2 guardado en: {file_path_2}")
            # fig2.show()
        else:
            print("No se pudo generar el gráfico detallado.")

        # Gráfico 3: Comparativa Agregada (2 barras)
        fig3 = plot_comparativa_boroughs_agregado(df_traffic, df_demand)
        if fig3:
            file_path_3 = output_dir / "comparativa_boroughs_agregado.html"
            fig3.write_html(file_path_3)
            print(f"Gráfico 3 guardado en: {file_path_3}")
            # fig3.show() 
        else:
            print("No se pudo generar el gráfico agregado.")

    else:
        print("No se pudieron cargar todos los datos.")

if __name__ == "__main__":
    main()