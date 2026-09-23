import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "datos"
RUTA_LIMPIOS = DATA_DIR / "limpios"
RUTA_GRAFOS = BASE_DIR / "graphs"


def cargar_dataset(path):
    return pd.read_parquet(path)


def calcular_impacto_por_tipo(df):
    
    impacto = (
        df[df["Event Type"].notna()]
        .groupby("Event Type")["pct_change_vs_baseline"]
        .agg(["mean", "median", "max", "count"])
        .sort_values("mean", ascending=False)
    )
    
    return impacto

def calcular_impacto_por_evento(df):
    
    event_summary = (
        df[df["event_id"].notna()]
        .groupby("event_id")["pct_change_vs_baseline"]
        .agg(
            count='size',
            mean_impact='mean',
            median_impact='median',
            max_impact='max'
        )
        .reset_index()
    )
    
    nombres = df[["event_id", "Event Name"]].drop_duplicates()
    
    event_summary = event_summary.merge(
        nombres,
        on="event_id",
        how="left"
    )
    
    return event_summary


def graficar_top_eventos(event_summary):
    
    top_events = event_summary.sort_values(
        "max_impact", ascending=False
    ).head(20)
    
    plt.figure(figsize=(12,6))
    sns.barplot(
        x='max_impact',
        y='Event Name',
        data=top_events,
        palette="viridis"
    )
    
    plt.xlabel("Máximo cambio de tráfico (%)")
    plt.ylabel("Evento")
    plt.title("Top 20 eventos que más alteran el tráfico en NYC")
    plt.tight_layout()
    plt.savefig(RUTA_GRAFOS / "top_events_traffic.png")
    plt.close()


def graficar_boxplot_por_tipo(df):
    
    plt.figure(figsize=(12,6))
    
    sns.boxplot(
        x='pct_change_vs_baseline',
        y='Event Type',
        data=df[df["Event Type"].notna()],
        orient='h',
        palette="coolwarm"
    )
    
    plt.xlabel("Cambio % vs tráfico base")
    plt.ylabel("Tipo de evento")
    plt.title("Distribución de cambios de tráfico por tipo de evento")
    plt.tight_layout()
    plt.savefig(RUTA_GRAFOS / "boxplot_event_type_traffic.png")
    plt.close()


def analizar_dia_especifico(df, fecha_str, file_name=None, boro=None):
    
    fecha = pd.to_datetime(fecha_str).date()
    day_data = df[df["timestamp"].dt.date == fecha]
    
    if boro:
        day_data = day_data[day_data["Boro"] == boro]
    
    if day_data.empty:
        print("No hay datos para ese día.")
        return
    
    resumen = day_data.groupby("hora_entera").agg({
        "Vol": "mean",
        "baseline_vol": "mean"
    })
    
    plt.figure(figsize=(12,6))
    
    plt.plot(resumen.index, resumen["Vol"], label="Tráfico real")
    plt.plot(resumen.index, resumen["baseline_vol"],
             linestyle="--", label="Baseline esperado")
    
    plt.xlabel("Hora del día")
    plt.ylabel("Volumen tráfico")
    plt.title(f"Tráfico vs Baseline - {fecha_str}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUTA_GRAFOS / f"comparacion_{file_name}.png")
    plt.close()

def graficar_distribucion_evento_vs_no_evento(df):

    df_plot = df.copy()
    df_plot["Tiene Evento"] = df_plot["Event Type"].notna()

    plt.figure(figsize=(10,6))
    
    sns.kdeplot(
        data=df_plot,
        x="pct_change_vs_baseline",
        hue="Tiene Evento",
        fill=True,
        common_norm=False
    )

    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("Cambio % vs baseline")
    plt.title("Distribución del cambio de tráfico: Con evento vs Sin evento")
    plt.tight_layout()
    plt.savefig(RUTA_GRAFOS / "distribucion_evento_vs_no_evento.png")
    plt.close()


def graficar_heatmap_por_tipo_hora(df):

    df_eventos = df[df["Event Type"].notna()].copy()

    heatmap_data = (
        df_eventos
        .groupby(["Event Type", "hora_entera"])["pct_change_vs_baseline"]
        .mean()
        .unstack()
    )

    plt.figure(figsize=(14,8))
    sns.heatmap(heatmap_data, cmap="coolwarm", center=0)
    plt.title("Impacto medio por tipo de evento y hora del día")
    plt.xlabel("Hora del día")
    plt.ylabel("Tipo de evento")
    plt.tight_layout()
    plt.savefig(RUTA_GRAFOS / "heatmap_tipo_hora.png")
    plt.close()


def eventos_mas_significativos(event_summary):

    top_positivos = (
        event_summary
        .sort_values("max_impact", ascending=False)
        .head(50)
    )

    top_negativos = (
        event_summary
        .sort_values("max_impact", ascending=True)
        .head(10)
    )

    print("\n🔺 TOP 50 EVENTOS CON MAYOR AUMENTO DE TRÁFICO")
    print(top_positivos[["Event Name", "max_impact"]])

    print("\n🔻 TOP 10 EVENTOS CON MAYOR CAÍDA DE TRÁFICO")
    print(top_negativos[["Event Name", "max_impact"]])

    return top_positivos, top_negativos

def graficar_eventos_extremos(top_positivos, top_negativos):

    import matplotlib.pyplot as plt

    extremos = pd.concat([top_positivos, top_negativos])

    extremos = extremos.sort_values("max_impact")

    plt.figure(figsize=(14, 12))

    plt.barh(
        extremos["Event Name"],
        extremos["max_impact"]
    )

    plt.axvline(0, linestyle="--")

    plt.xlabel("Cambio máximo % vs baseline")
    plt.ylabel("Evento")
    plt.title("Eventos con mayor alteración del tráfico (positivos y negativos)")

    plt.tight_layout()
    plt.savefig(RUTA_GRAFOS / "eventos_extremos_impacto.png")
    plt.close()

def main():
    

    print("Cargando dataset transformado...")
    df = cargar_dataset(RUTA_LIMPIOS / "traffic_eventos_transformado.parquet")
    
    print("Calculando impacto por tipo...")
    impacto_tipo = calcular_impacto_por_tipo(df)
    print(impacto_tipo)
    
    print("Calculando impacto por evento individual...")
    event_summary = calcular_impacto_por_evento(df)
    
    print("Generando gráficas...")
    graficar_top_eventos(event_summary)
    graficar_boxplot_por_tipo(df)
    
    print("Analizando días específicos...")
    analizar_dia_especifico(df, "2023-11-05", "NYC_Marathon")
    analizar_dia_especifico(df, "2023-02-04", "Chinese New Year")
    analizar_dia_especifico(df, "2023-07-04", "4th July")

    print("Calculando eventos más significativos...")
    top_pos, top_neg = eventos_mas_significativos(event_summary)

    print("Generando gráfica de eventos extremos...")
    graficar_eventos_extremos(top_pos, top_neg)


    graficar_distribucion_evento_vs_no_evento(df)
    graficar_heatmap_por_tipo_hora(df)
    print("Análisis completado.")


# ------------------------------------------------------------
if __name__ == "__main__":
    main()
