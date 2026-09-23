import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv, find_dotenv

def load_data():
    """Intenta cargar desde MinIO, si falla, carga localmente."""
    load_dotenv(find_dotenv())
    
    # Datos de MinIO
    minio_endpoint = os.getenv("MINIO_ENDPOINT")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    bucket = os.getenv("MINIO_BUCKET")
    group_path = os.getenv("MINIO_GROUP_PATH")

    # Ruta S3 (con /limpios/)
    ruta_s3 = f"s3://{bucket}/{group_path}/limpios/resumen_zona_hora.parquet"
    
    # Ruta local dinámica basada en la estructura del repo
    project_root = Path(__file__).resolve().parents[2]
    ruta_local = project_root / "datos" / "limpios" / "resumen_zona_hora.parquet"

    try:
        print(f"Intentando descargar datos desde MinIO: {ruta_s3}")
        df = pd.read_parquet(
            ruta_s3,
            storage_options={
                "key": access_key,
                "secret": secret_key,
                "client_kwargs": {'endpoint_url': minio_endpoint}
            }
        )
    except Exception as e:
        print(f"Fallo de conexión con MinIO: {str(e).splitlines()[0]}")
        print(f"Hacemos fallback y leemos localmente desde: {ruta_local}")
        df = pd.read_parquet(ruta_local)

    return df

def train_baseline(df):
    """Entrena un modelo base usando Scikit-Learn con variables exógenas"""
    print("\nPreparando el pipeline de Scikit-Learn...")

    # Aseguramos el día de la semana
    if 'day_of_week' not in df.columns and 'date_only' in df.columns:
        df['day_of_week'] = pd.to_datetime(df['date_only']).dt.dayofweek

    # --- CAMBIO: Añadimos dinámicamente las variables exógenas ---
    features = ['pulocationid', 'pickup_hour', 'day_of_week']
    exogenas = [
        "temperature_2m", "precipitation", "snowfall", "hay_evento", 
        "num_restaurantes", "precio_medio_rest", "num_alquileres", "precio_medio_alquiler"
    ]
    
    # Solo añadimos las que realmente existan en el parquet
    for col in exogenas:
        if col in df.columns:
            features.append(col)
            
    print(f"Variables utilizadas para entrenar: {features}")

    # Rellenamos nulos por seguridad
    df = df.fillna(0)

    X = df[features]
    y = df['demanda_viajes']

    # Dividimos en Train y Test (80% - 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocesamiento: OHE para ID de zona y día de la semana. 
    # remainder='passthrough' deja pasar el clima y restaurantes intactos.
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['pulocationid', 'day_of_week'])
        ],
        remainder='passthrough'
    )

    # Árbol de decisión simple como Baseline
    baseline_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(max_depth=10, random_state=42))
    ])

    print("Entrenando Árbol de Decisión (Baseline enriquecido)...")
    baseline_model.fit(X_train, y_train)

    # Evaluación
    y_pred = baseline_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print("-" * 40)
    print(" RESULTADOS DEL BASELINE (Scikit-Learn)")
    print(f" RMSE: {rmse:.2f} viajes")
    print(f" MAE:  {mae:.2f} viajes")
    print("-" * 40)

    return baseline_model

if __name__ == "__main__":
    df_data = load_data()
    print(f"\nFilas procesadas: {df_data.shape[0]}")
    model = train_baseline(df_data)