import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ===============================
#  Rutas del proyecto
# ===============================

BASE_DIR = Path(__file__).resolve()
PROJECT_ROOT = BASE_DIR.parents[3]  # ajusta si tu estructura cambia

# Ruta de CARGA
DATA_PATH = PROJECT_ROOT / "datos" / "limpios" / "hourly_aggregate.parquet"

# Ruta de DESTINO
VIS_ROOT = BASE_DIR.parents[0]
OUTPUT_PATH = VIS_ROOT / "YLC_FHV_clima"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# ==============================
# CARGA DE DATOS
# ==============================

def cargar_datos(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.reset_index()

    # Asegurar formato datetime
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"])

    # Crear variable binaria lluvia
    df["rain_flag"] = np.where(df["precipitation"] > 0, "Rain", "No Rain")

    # Extraer hora del día
    df["hour"] = df["datetime_hour"].dt.hour

    return df


# ==============================
# IMPACTO PORCENTUAL LLUVIA
# ==============================

def grafico_impacto_lluvia(df: pd.DataFrame, output_dir: Path):
    rain_mean = df.groupby("rain_flag")[["YLC", "FHV", "total", "market_share"]].mean()

    impact = ((rain_mean.loc["Rain"] - rain_mean.loc["No Rain"])
              / rain_mean.loc["No Rain"]) * 100

    impact_df = impact.reset_index()
    impact_df.columns = ["variable", "percent_change"]

    fig = px.bar(
        impact_df,
        x="variable",
        y="percent_change",
        title="Impacto porcentual de la lluvia",
        text="percent_change"
    )

    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")

    fig.write_html(output_dir / "impacto_lluvia.html")


# ==============================
# MAIN
# ==============================

def main():
    print("Cargando datos...")
    df = cargar_datos(DATA_PATH)


    print("Generando visualizaciones...")

    grafico_impacto_lluvia(df, OUTPUT_PATH)

    print("Visualizaciones generadas correctamente.")
    print(f"Archivos guardados en: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
