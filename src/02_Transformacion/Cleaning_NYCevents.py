import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "datos"
CRUDOS_DIR = DATA_DIR / "crudos"
LIMPIOS_DIR = DATA_DIR / "limpios"

def filtrar_mlb_primera_semana(input_path: Path):
    
    df = pd.read_csv(input_path)

    df["start_time_ny"] = pd.to_datetime(df["start_time_ny"], errors="coerce")
    df["end_time_ny"] = pd.to_datetime(df["end_time_ny"], errors="coerce")

    df_first_week = df[df["start_time_ny"].dt.day.between(14)].copy()
    df_first_week.sort_values("start_time_ny", inplace=True)

    print(f"Partidos en la primera semana de cada mes: {len(df_first_week)}")
    print(df_first_week[["venue", "start_time_ny", "end_time_ny"]].head())

    df_first_week.to_csv(LIMPIOS_DIR / "mlb_nyc_first_week.csv", index=False)

    return df_first_week

def cargar_datasets(eventos_path, trafico_path):
    df_eventos = pd.read_parquet(eventos_path)
    df_trafico = pd.read_csv(trafico_path)
    return df_eventos, df_trafico


def limpiar_eventos(df):
    
    columnas_eliminar = [
        "Event Agency",
        "Police Precinct",
        "Event Street Side",
        "Community Board"
    ]
    
    df = df.drop(columns=columnas_eliminar, errors="ignore")
    
    tipos_excluir = [
        "Sport - Youth",
        "Sport - Adult",
        "Special Event"
    ]
    
    df = df[~df["Event Type"].isin(tipos_excluir)]
    
    return df


def preparar_fechas(df_eventos, df_trafico):
    
    df_trafico["timestamp"] = pd.to_datetime(df_trafico["timestamp"])
    df_eventos["Start Date/Time"] = pd.to_datetime(df_eventos["Start Date/Time"])
    df_eventos["End Date/Time"] = pd.to_datetime(df_eventos["End Date/Time"])
    
    return df_eventos, df_trafico


def calcular_baseline(df_trafico):
    
    baseline = (
        df_trafico
        .groupby(["Boro", "dia_semana", "hora_entera"])["Vol"]
        .mean()
        .reset_index()
        .rename(columns={"Vol": "baseline_vol"})
    )
    
    df_trafico = df_trafico.merge(
        baseline,
        on=["Boro", "dia_semana", "hora_entera"],
        how="left"
    )
    
    return df_trafico

def expandir_eventos_por_hora(df_eventos):
    
    expanded_events = []
    
    for _, row in df_eventos.iterrows():
        
        hours = pd.date_range(
            start=row["Start Date/Time"],
            end=row["End Date/Time"],
            freq="h"
        )
        
        for h in hours:
            expanded_events.append({
                "timestamp": h,
                "Boro": row["Event Borough"],
                "Event Type": row["Event Type"],
                "event_id": row["Event ID"],
                "Event Name": row["Event Name"]
            })
    
    return pd.DataFrame(expanded_events)


def integrar_eventos_trafico(df_trafico, events_hourly):
    
    df = df_trafico.merge(
        events_hourly,
        on=["timestamp", "Boro"],
        how="left"
    )
    
    df["pct_change_vs_baseline"] = (
        (df["Vol"] - df["baseline_vol"]) /
        df["baseline_vol"]
    ) * 100
    
    return df


def cargar_eventos_mlb(path_csv, id_offset):
    
    mlb = pd.read_csv(path_csv)
    
    mlb["start_time_ny"] = pd.to_datetime(mlb["start_time_ny"])
    mlb["end_time_ny"] = pd.to_datetime(mlb["end_time_ny"])
    
    mlb["start_time_ny"] = mlb["start_time_ny"].dt.tz_localize(None)
    mlb["end_time_ny"] = mlb["end_time_ny"].dt.tz_localize(None)

    venue_to_boro = {
        "Yankee Stadium": "Bronx",
        "Citi Field": "Queens"
    }
    
    mlb["Event Borough"] = mlb["venue"].map(venue_to_boro)
    mlb_ids = range(id_offset + 1, id_offset + 1 + len(mlb))
    
    mlb_formatted = pd.DataFrame({
        "Event ID": list(mlb_ids),
        "Event Name": mlb["venue"] + " MLB Game",
        "Event Type": "MLB",
        "Event Borough": mlb["Event Borough"],
        "Start Date/Time": mlb["start_time_ny"],
        "End Date/Time": mlb["end_time_ny"]
    })
    
    return mlb_formatted


def main():
    
    eventos_path = CRUDOS_DIR / "NYC_events_2023_first_half.parquet"
    trafico_path = CRUDOS_DIR / "dataset_trafico_vis_ready.csv"
    
    print("Cargando datasets...")
    df_eventos, df_trafico = cargar_datasets(eventos_path, trafico_path)
    
    print("Limpiando eventos...")
    df_eventos = limpiar_eventos(df_eventos)

    print("Guardando eventos limpios")
    df_eventos.to_parquet(LIMPIOS_DIR / "eventos_limpios.parquet", index=False)
    
    print("Preparando fechas...")
    df_eventos, df_trafico = preparar_fechas(df_eventos, df_trafico)
    
    print("Calculando baseline...")
    df_trafico = calcular_baseline(df_trafico)
    
    print("Expandiendo eventos por hora...")
    events_hourly = expandir_eventos_por_hora(df_eventos)
    
    print("Integrando eventos con tráfico...")
    df_final = integrar_eventos_trafico(df_trafico, events_hourly)

    max_existing_id = df_eventos["Event ID"].max()
    print("Cargando MLB...")
    filtrar_mlb_primera_semana(CRUDOS_DIR / "mlb_games_ny_stadiums_2023.csv")
    mlb_df = cargar_eventos_mlb(CRUDOS_DIR / "mlb_nyc_first_week.csv", max_existing_id)

    mlb_hourly = expandir_eventos_por_hora(mlb_df)
    df_final["timestamp"] = df_final["timestamp"].dt.floor("h")
    mlb_hourly["timestamp"] = mlb_hourly["timestamp"].dt.floor("h")

    print("Integrando MLB después del baseline...")
    df_final = df_final.merge(
        mlb_hourly,
        on=["timestamp", "Boro"],
        how="left",
        suffixes=("", "_mlb")
    )

    df_final["Event Type"] = df_final["Event Type"].fillna(df_final["Event Type_mlb"])
    df_final["Event Name"] = df_final["Event Name"].fillna(df_final["Event Name_mlb"])
    df_final["event_id"] = df_final["event_id"].fillna(df_final["event_id_mlb"])

    df_final.drop(columns=[
        "Event Type_mlb",
        "Event Name_mlb",
        "event_id_mlb"
    ], inplace=True)
    
    print("Tipos de evento únicos en traffic después del merge:")
    print(df_final["Event Type"].dropna().unique())

    mlb_count = df_final[df_final["Event Type"] == "MLB"].shape[0]
    print(f"Número de filas asociadas a MLB: {mlb_count}")

    print(df_final[df_final["Event Type"] == "MLB"].head())

    print("Guardando dataset transformado...")
    df_final.to_parquet(LIMPIOS_DIR / "traffic_eventos_transformado.parquet", index=False)
    
    print("Proceso completado correctamente.")


if __name__ == "__main__":
    main()
