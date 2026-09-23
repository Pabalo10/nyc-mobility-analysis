from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# ==========================
# CONFIG FASE 2
# ==========================

RENTA_PATH = Path("datos/limpios/rentabilidad_historica_fase2.parquet")
DEMANDA_PATH = Path("datos/limpios/demandas_base_fase2.parquet")
DATASET_FINAL_PATH = Path("datos/limpios/dataset_entrenamiento_final_fase2.parquet")
TRAFICO_PATH = Path("datos/limpios/dataset_trafico_vis_ready.parquet")
TAXI_ZONES_SHP = Path("datos/limpios/taxi_zones.shp")

MODEL_PATH = Path("src/07_despliegue/models/rf_model_fase2.pkl")
TRAIN_PATH = Path("datos/limpios/data_train_fase2.parquet")
TEST_PATH = Path("datos/limpios/data_test_fase2.parquet")

TARGET_COL = "rentabilidad_score"
JOIN_KEYS = ["pulocationid", "day_of_week", "pickup_hour"]
TEST_SIZE = 0.2
SEED = 42
CV_FOLDS = 5

DAY_MAP_EN = {
    "monday": 1,
    "tuesday": 2,
    "wednesday": 3,
    "thursday": 4,
    "friday": 5,
    "saturday": 6,
    "sunday": 7,
}

DAY_MAP_ES = {
    "lunes": 1,
    "martes": 2,
    "miercoles": 3,
    "miércoles": 3,
    "jueves": 4,
    "viernes": 5,
    "sabado": 6,
    "sábado": 6,
    "domingo": 7,
}


def leer_parquet_robusto(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")
    if path.is_dir():
        archivos = sorted(path.glob("*.parquet"))
        if not archivos:
            raise FileNotFoundError(f"No hay .parquet válidos dentro de: {path}")
        return pd.read_parquet([str(p) for p in archivos])
    return pd.read_parquet(path)


def imprimir_metadata(nombre: str, path: Path, df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print(f"METADATA -> {nombre}")
    print(f"Ruta: {path}")
    print(f"Shape: {df.shape}")
    print(f"Columnas: {list(df.columns)}")
    print("Dtypes:", {c: str(t) for c, t in df.dtypes.items()})
    nulls = df.isna().sum()
    null_cols = {c: int(v) for c, v in nulls.items() if int(v) > 0}
    print("Nulos:", null_cols if null_cols else "{}")


def validar_columnas(df: pd.DataFrame, cols: list, nombre: str) -> None:
    faltantes = [c for c in cols if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en {nombre}: {faltantes}")


def normalizar_day_of_week(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("Int64")
    norm = s.astype(str).str.strip().str.lower()
    mapped = norm.map(DAY_MAP_EN).fillna(norm.map(DAY_MAP_ES))
    return mapped.astype("Int64")


def normalizar_borough(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.title()


def construir_mapa_zona_borough(shp_path: Path) -> pd.DataFrame:
    if not shp_path.exists():
        raise FileNotFoundError(f"No existe shapefile de taxi zones: {shp_path}")

    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("Falta geopandas para mapear pulocationid -> borough") from exc

    gdf = gpd.read_file(shp_path)
    validar_columnas(gdf, ["LocationID", "borough"], "taxi_zones.shp")
    df_map = gdf[["LocationID", "borough"]].copy()
    df_map = df_map.rename(columns={"LocationID": "pulocationid"})
    df_map["pulocationid"] = pd.to_numeric(df_map["pulocationid"], errors="coerce").astype("Int64")
    df_map["borough"] = normalizar_borough(df_map["borough"])
    return df_map.dropna(subset=["pulocationid"]).drop_duplicates("pulocationid")


def preparar_trafico(df_trafico: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    df = df_trafico.copy()
    validar_columnas(df, ["hora_entera", "Vol"], "dataset_trafico_vis_ready")

    df = df.rename(columns={"hora_entera": "pickup_hour", "Vol": "traffic_total"})
    df["pickup_hour"] = pd.to_numeric(df["pickup_hour"], errors="coerce").astype("Int64")

    if "dia_semana" in df.columns:
        df["day_of_week"] = normalizar_day_of_week(df["dia_semana"])
    elif "timestamp" in df.columns:
        dt = pd.to_datetime(df["timestamp"], errors="coerce")
        df["day_of_week"] = dt.dt.dayofweek + 1
    else:
        df["day_of_week"] = pd.NA

    if "Boro" in df.columns:
        df["borough"] = normalizar_borough(df["Boro"])
        join_keys = ["borough", "pickup_hour", "day_of_week"]
        print("✅ Tráfico con borough detectado: se agrega por borough+hora+day_of_week.")
    else:
        join_keys = ["pickup_hour", "day_of_week"]
        print("⚠️ Tráfico sin borough: se agrega por hora+day_of_week.")

    df = df.dropna(subset=join_keys + ["traffic_total"])
    df = df.groupby(join_keys, as_index=False)["traffic_total"].mean()

    traffic_min = df["traffic_total"].min()
    traffic_max = df["traffic_total"].max()
    if pd.isna(traffic_min) or pd.isna(traffic_max) or traffic_max <= traffic_min:
        df["traffic_norm"] = 0.5
    else:
        df["traffic_norm"] = (df["traffic_total"] - traffic_min) / (traffic_max - traffic_min)

    return df, join_keys


def add_manual_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pickup_hour_sq"] = out["pickup_hour"].astype(float) ** 2

    # Interacciones hora x borough para captar patrones horarios por distrito.
    if "borough" in out.columns:
        boro_dummies = pd.get_dummies(out["borough"], prefix="boro", dtype="int8")
        for col in boro_dummies.columns:
            out[f"hour_x_{col}"] = out["pickup_hour"].astype(float) * boro_dummies[col].astype(float)

    return out


def fit_target_encoding_map(train_df: pd.DataFrame, cat_col: str, target_col: str, alpha: float = 20.0):
    global_mean = train_df[target_col].mean()
    stats = train_df.groupby(cat_col)[target_col].agg(["mean", "count"]).reset_index()
    stats["te"] = (stats["count"] * stats["mean"] + alpha * global_mean) / (stats["count"] + alpha)
    te_map = dict(zip(stats[cat_col], stats["te"]))
    return te_map, global_mean


def apply_target_encoding(
    df: pd.DataFrame, cat_col: str, te_map: dict, global_mean: float, new_col: str
) -> pd.DataFrame:
    out = df.copy()
    out[new_col] = out[cat_col].map(te_map).fillna(global_mean)
    return out


def clip_target_on_train_quantiles(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    q_low: float = 0.01,
    q_high: float = 0.99,
):
    low = train_df[target_col].quantile(q_low)
    high = train_df[target_col].quantile(q_high)

    tr = train_df.copy()
    te = test_df.copy()

    tr[target_col] = tr[target_col].clip(lower=low, upper=high)
    te[target_col] = te[target_col].clip(lower=low, upper=high)

    print(f"🎯 Clip target por cuantiles train: low={low:.4f}, high={high:.4f}")
    return tr, te


def main():
    print("📦 Cargando datasets Fase 2 obligatorios...")
    df_renta = leer_parquet_robusto(RENTA_PATH)
    df_demanda = leer_parquet_robusto(DEMANDA_PATH)
    df_final = leer_parquet_robusto(DATASET_FINAL_PATH)
    df_trafico_raw = leer_parquet_robusto(TRAFICO_PATH)

    imprimir_metadata("rentabilidad_historica_fase2", RENTA_PATH, df_renta)
    imprimir_metadata("demandas_base_fase2", DEMANDA_PATH, df_demanda)
    imprimir_metadata("dataset_entrenamiento_final_fase2", DATASET_FINAL_PATH, df_final)
    imprimir_metadata("dataset_trafico_vis_ready (trafico)", TRAFICO_PATH, df_trafico_raw)

    validar_columnas(df_renta, JOIN_KEYS + [TARGET_COL], "rentabilidad_historica_fase2")
    validar_columnas(df_demanda, JOIN_KEYS + ["demanda_predicha"], "demandas_base_fase2")
    validar_columnas(df_final, JOIN_KEYS + [TARGET_COL, "demanda_predicha"], "dataset_entrenamiento_final_fase2")

    print("\n🔗 Armando dataset base con rentabilidad + demanda...")
    df_model = df_renta.merge(
        df_demanda[JOIN_KEYS + ["demanda_predicha"]],
        on=JOIN_KEYS,
        how="inner",
    )
    print(f"Filas tras merge renta+demanda: {len(df_model)}")

    print("🧪 Cruzando con dataset_entrenamiento_final_fase2 para usar los 3 artefactos...")
    keys_final = df_final[JOIN_KEYS].drop_duplicates()
    df_model = df_model.merge(keys_final, on=JOIN_KEYS, how="inner")
    print(f"Filas en intersección de los 3 datasets: {len(df_model)}")

    print("🗺️ Mapeando pulocationid -> borough con taxi_zones.shp...")
    df_zone_borough = construir_mapa_zona_borough(TAXI_ZONES_SHP)
    df_model = df_model.merge(df_zone_borough, on="pulocationid", how="left")

    print("🚦 Agregando features de tráfico...")
    df_trafico, traffic_join_keys = preparar_trafico(df_trafico_raw)
    df_model = df_model.merge(df_trafico, on=traffic_join_keys, how="left")

    for col in ["traffic_total", "traffic_norm"]:
        if col in df_model.columns:
            med = df_model[col].median()
            df_model[col] = df_model[col].fillna(med if not pd.isna(med) else 0.0)

    print("\n🧹 Limpieza final...")
    df_model = df_model.dropna(subset=JOIN_KEYS + [TARGET_COL, "demanda_predicha"])
    df_model = df_model[(df_model[TARGET_COL] > 0) & (df_model[TARGET_COL] < 15.0)]
    print(f"Shape para modelado: {df_model.shape}")

    print("\n✂️ Dividiendo en data_train / data_test...")
    base_train_df, base_test_df = train_test_split(
        df_model,
        test_size=TEST_SIZE,
        random_state=SEED,
    )

    base_train_df, base_test_df = clip_target_on_train_quantiles(
        base_train_df,
        base_test_df,
        TARGET_COL,
        q_low=0.01,
        q_high=0.99,
    )

    base_train_df = add_manual_features(base_train_df)
    base_test_df = add_manual_features(base_test_df)

    te_map, te_global = fit_target_encoding_map(
        base_train_df,
        cat_col="pulocationid",
        target_col=TARGET_COL,
        alpha=20.0,
    )
    base_train_df = apply_target_encoding(
        base_train_df,
        cat_col="pulocationid",
        te_map=te_map,
        global_mean=te_global,
        new_col="pulocationid_te",
    )
    base_test_df = apply_target_encoding(
        base_test_df,
        cat_col="pulocationid",
        te_map=te_map,
        global_mean=te_global,
        new_col="pulocationid_te",
    )

    base_train_df.to_parquet(TRAIN_PATH, index=False)
    base_test_df.to_parquet(TEST_PATH, index=False)
    print(f"Train guardado en: {TRAIN_PATH} -> {base_train_df.shape}")
    print(f"Test guardado en: {TEST_PATH} -> {base_test_df.shape}")

    print("\n⚙️ Construyendo features y entrenando benchmark...")
    cat_cols = ["day_of_week"]
    if "borough" in base_train_df.columns:
        cat_cols.append("borough")

    train_enc = pd.get_dummies(base_train_df, columns=cat_cols, dtype="int8")
    test_enc = pd.get_dummies(base_test_df, columns=cat_cols, dtype="int8")
    train_enc, test_enc = train_enc.align(test_enc, join="left", axis=1, fill_value=0)

    drop_cols = [TARGET_COL, "pulocationid"]
    feature_cols = [c for c in train_enc.columns if c not in drop_cols]

    X_train = train_enc[feature_cols]
    y_train = train_enc[TARGET_COL]
    X_test = test_enc[feature_cols]
    y_test = test_enc[TARGET_COL]

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scorers = {"mae": "neg_mean_absolute_error", "r2": "r2"}

    print("\n🔎 Iniciando benchmark con CV y tuning...")

    candidatos = {
        "RandomForest": {
            "estimator": RandomForestRegressor(random_state=SEED, n_jobs=-1),
            "params": {
                "n_estimators": [200, 300, 400],
                "max_depth": [None, 10, 16, 24],
                "min_samples_split": [2, 4, 8],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", 0.6, 0.8],
            },
            "n_iter": 16,
        },
        "ExtraTrees": {
            "estimator": ExtraTreesRegressor(random_state=SEED, n_jobs=-1),
            "params": {
                "n_estimators": [200, 300, 400],
                "max_depth": [None, 10, 16, 24],
                "min_samples_split": [2, 4, 8],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", 0.6, 0.8],
            },
            "n_iter": 16,
        },
        "HistGBR": {
            "estimator": HistGradientBoostingRegressor(random_state=SEED),
            "params": {
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [None, 6, 10, 14],
                "max_leaf_nodes": [15, 31, 63],
                "min_samples_leaf": [20, 40, 60],
                "l2_regularization": [0.0, 0.01, 0.1],
            },
            "n_iter": 12,
        },
        "Ridge": {
            "estimator": Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("model", Ridge(random_state=SEED)),
            ]),
            "params": {
                "model__alpha": [0.01, 0.1, 1.0, 10.0, 30.0, 100.0],
            },
            "n_iter": 6,
        },
    }

    resultados = []
    mejor_nombre = None
    mejor_search = None
    mejor_cv_mae = float("inf")

    for nombre, cfg in candidatos.items():
        print(f"\n▶ Modelo: {nombre}")
        search = RandomizedSearchCV(
            estimator=cfg["estimator"],
            param_distributions=cfg["params"],
            n_iter=cfg["n_iter"],
            scoring=scorers,
            refit="mae",
            cv=cv,
            n_jobs=-1,
            random_state=SEED,
            verbose=0,
        )
        search.fit(X_train, y_train)

        cv_mae = -search.best_score_
        best_idx = search.best_index_
        cv_r2 = search.cv_results_["mean_test_r2"][best_idx]

        resultados.append(
            {
                "modelo": nombre,
                "cv_mae": cv_mae,
                "cv_r2": cv_r2,
                "best_params": search.best_params_,
            }
        )

        print(f"CV MAE: {cv_mae:.4f} | CV R²: {cv_r2:.4f}")
        print(f"Best params: {search.best_params_}")

        if cv_mae < mejor_cv_mae:
            mejor_cv_mae = cv_mae
            mejor_nombre = nombre
            mejor_search = search

    resultados_df = pd.DataFrame(resultados).sort_values("cv_mae", ascending=True)
    print("\n🏆 Ranking de modelos (CV):")
    print(resultados_df[["modelo", "cv_mae", "cv_r2"]].to_string(index=False))

    mejor_modelo = mejor_search.best_estimator_
    pred = mejor_modelo.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print("\n📈 Métricas Holdout (modelo ganador)")
    print(f"Modelo ganador (CV): {mejor_nombre}")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")
    print(f"Target usado: {TARGET_COL}")
    print(f"Cantidad de features: {X_train.shape[1]}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mejor_modelo, MODEL_PATH)
    print(f"💾 Mejor modelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()