"""
aldani2_lib.py — Integración de los modelos v8 (compañero, 6 jul) al dashboard.py
existente. Adaptación CSV (sin SQLite) de src/predict.py y src/features.py del
paquete "ALDIMI Gestión Predictiva v8" que llegó comprimido en Aldani2.rar.

Dos modelos:
  1. modelo_costos_producto.pkl   — ExtraTreesRegressor. Predice solo la cantidad
     requerida la próxima semana por producto (catálogo nutricional de 36
     productos, distinto al catálogo operativo anterior). El costo se calcula
     después: cantidad_predicha * costo_unitario_promedio_reciente. También
     genera nivel de riesgo, días de cobertura y una recomendación en texto.
  2. modelo_prioridad_albergue.pkl — XGBClassifier multiclase ponderado
     (baja/media/alta/urgente) para priorizar cupos de albergue. Devuelve
     probabilidades, puntaje de prioridad (0-100) y motivos de la decisión.

No se usa la base SQLite del compañero (mismo motivo que con aldimi_core.db:
evitar I/O de SQLite sobre la ruta montada de Windows). Los datos que el
modelo de productos necesita ya vienen precalculados por semana en
v8_ml_dataset_costos_producto.csv, así que basta con tomar la última semana
por producto — no hace falta SQLite ni recalcular features desde cero.
"""
from __future__ import annotations

import os
import warnings
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Compatibilidad scikit-learn (mismo parche que ya usa dashboard.py para el
# modelo de pacientes binario; estos 2 modelos nuevos lo necesitan también).
try:
    from sklearn.compose import _column_transformer as _ct_mod

    if not hasattr(_ct_mod, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        _ct_mod._RemainderColsList = _RemainderColsList
except Exception:
    pass


def _base_dir() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


BASE = _base_dir()
V8_PRODUCTOS_CSV = os.path.join(BASE, "data", "processed", "v8_productos.csv")
V8_CATEGORIAS_CSV = os.path.join(BASE, "data", "processed", "v8_categorias_producto.csv")
V8_ML_COSTOS_CSV = os.path.join(BASE, "data", "processed", "v8_ml_dataset_costos_producto.csv")
V8_GEO_CSV = os.path.join(BASE, "data", "processed", "v8_geo_departamentos.csv")
V8_MOVIMIENTOS_CSV = os.path.join(BASE, "data", "processed", "v8_movimientos_inventario.csv")

MODEL_COSTOS_PATH = os.path.join(BASE, "models", "modelo_costos_producto.pkl")
MODEL_PRIORIDAD_PATH = os.path.join(BASE, "models", "modelo_prioridad_albergue.pkl")

FAMILIAS_BASE_ENTRENAMIENTO = 50  # el histórico de entrenamiento representa 50 familias


# ══════════════════════════════════════════════════════════════════════════
# Catálogo y datos base (CSV, cacheado por el llamador con @st.cache_data)
# ══════════════════════════════════════════════════════════════════════════
def cargar_catalogo_v8() -> pd.DataFrame:
    productos = pd.read_csv(V8_PRODUCTOS_CSV)
    categorias = pd.read_csv(V8_CATEGORIAS_CSV)
    out = productos.merge(categorias, on="id_categoria", how="left")
    return out[out["activo"] == 1].sort_values(["nombre_categoria", "nombre_producto"])


def cargar_geo_v8() -> dict[str, int]:
    geo = pd.read_csv(V8_GEO_CSV)
    return {str(r["departamento"]): int(r["distancia_lima_km"]) for _, r in geo.iterrows()}


def latest_product_feature_rows_v8() -> pd.DataFrame:
    """Última semana disponible por producto en el dataset de features del compañero."""
    df = pd.read_csv(V8_ML_COSTOS_CSV)
    idx = df.groupby("id_producto")["id_semana"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def cargar_movimientos_v8() -> pd.DataFrame:
    cols = ["fecha", "id_producto", "cantidad", "nota"]
    if os.path.exists(V8_MOVIMIENTOS_CSV):
        return pd.read_csv(V8_MOVIMIENTOS_CSV)
    return pd.DataFrame(columns=cols)


def guardar_ingreso_v8(id_producto: int, cantidad: float, nota: str = "") -> None:
    df_mov = cargar_movimientos_v8()
    nueva = pd.DataFrame([{
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "id_producto": int(id_producto), "cantidad": cantidad, "nota": nota,
    }])
    os.makedirs(os.path.dirname(V8_MOVIMIENTOS_CSV), exist_ok=True)
    pd.concat([df_mov, nueva], ignore_index=True).to_csv(V8_MOVIMIENTOS_CSV, index=False)


def stock_actual_v8(latest_rows: pd.DataFrame) -> dict[int, float]:
    """stock_fin_semana histórico (≥0) + ingresos manuales registrados desde el dashboard."""
    base = {
        int(r["id_producto"]): max(float(r["stock_fin_semana"]), 0.0)
        for _, r in latest_rows.iterrows()
    }
    df_mov = cargar_movimientos_v8()
    if len(df_mov):
        extra = df_mov.groupby("id_producto")["cantidad"].sum()
        for pid, cantidad in extra.items():
            base[int(pid)] = base.get(int(pid), 0.0) + float(cantidad)
    return base


def build_forecast_frame(latest_rows: pd.DataFrame, stock_by_product: dict[int, float] | None = None) -> pd.DataFrame:
    """Copiado de src/features.py del paquete v8 (sin cambios de lógica)."""
    df = latest_rows.copy()
    if stock_by_product:
        df["stock_inicio_semana"] = df["id_producto"].map(stock_by_product).fillna(df["stock_inicio_semana"]).astype(float)
        df["stock_fin_semana"] = df["stock_inicio_semana"]
    df["demanda_no_cubierta"] = np.maximum(df["cantidad_necesaria"] - df["stock_inicio_semana"] - df["ingresos_semana"], 0)
    return df


def nivel_riesgo_producto(stock_actual: float, cantidad_predicha: float) -> tuple[str, float, str]:
    """Copiado de src/features.py del paquete v8 (sin cambios de lógica)."""
    demanda_diaria = max(float(cantidad_predicha) / 7.0, 0.01)
    dias_cobertura = float(stock_actual) / demanda_diaria
    if dias_cobertura < 3:
        return "crítico", round(dias_cobertura, 1), "Compra o donación inmediata; priorizar entrega controlada."
    if dias_cobertura < 7:
        return "alto", round(dias_cobertura, 1), "Programar reposición esta semana y revisar entregas pendientes."
    if dias_cobertura < 14:
        return "medio", round(dias_cobertura, 1), "Mantener seguimiento y preparar reposición preventiva."
    return "bajo", round(dias_cobertura, 1), "Stock suficiente; mantener monitoreo semanal."


# ══════════════════════════════════════════════════════════════════════════
# Predicción de productos (adaptado de src/predict.py del paquete v8)
# ══════════════════════════════════════════════════════════════════════════
def _load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el modelo: {path}")
    return joblib.load(path)


def predict_product_costs_v8(df_features: pd.DataFrame, factor_familias: float = 1.0) -> pd.DataFrame:
    bundle = _load_model(MODEL_COSTOS_PATH)
    features = bundle["features"]
    model = bundle["model"]

    out = df_features.copy()
    model_input = out.copy()
    for col in bundle.get("categorical_features", []):
        if col in model_input.columns:
            model_input[col] = model_input[col].astype(str)

    pred_qty = np.asarray(model.predict(model_input[features])).reshape(-1)
    pred_qty = pred_qty * float(factor_familias)
    out["pred_cantidad_siguiente_semana"] = np.maximum(pred_qty, 0).round(0)

    recent_col = bundle.get("recent_price_column", "rolling_precio_4sem")
    fallback_col = bundle.get("fallback_price_column", "precio_unitario_estimado")
    precio = pd.to_numeric(out.get(recent_col), errors="coerce") if recent_col in out.columns else pd.Series(np.nan, index=out.index)
    if fallback_col in out.columns:
        precio = precio.fillna(pd.to_numeric(out[fallback_col], errors="coerce"))
    precio = precio.fillna(0).clip(lower=0)

    out["costo_unitario_promedio_reciente"] = precio.round(2)
    out["pred_costo_siguiente_semana"] = (
        out["pred_cantidad_siguiente_semana"] * out["costo_unitario_promedio_reciente"]
    ).clip(lower=0).round(2)

    riesgos = out.apply(
        lambda r: nivel_riesgo_producto(r.get("stock_inicio_semana", 0), r["pred_cantidad_siguiente_semana"]),
        axis=1,
    )
    out["nivel_riesgo"] = [r[0] for r in riesgos]
    out["dias_cobertura"] = [r[1] for r in riesgos]
    out["recomendacion"] = [r[2] for r in riesgos]
    return out


def calcular_predicciones_productos_v8(factor_familias: float = 1.0) -> pd.DataFrame:
    latest = latest_product_feature_rows_v8()
    stocks = stock_actual_v8(latest)
    forecast = build_forecast_frame(latest, stocks)
    pred = predict_product_costs_v8(forecast, factor_familias=factor_familias)
    catalog = cargar_catalogo_v8()[[
        "id_producto", "codigo_producto", "nombre_producto", "nombre_categoria",
        "tipo_necesidad", "unidad_compra", "precio_referencia_unitario", "criticidad_producto",
    ]].copy()
    pred["id_producto"] = pd.to_numeric(pred["id_producto"], errors="coerce").astype("Int64")
    catalog["id_producto"] = pd.to_numeric(catalog["id_producto"], errors="coerce").astype("Int64")
    return pred.merge(catalog, on="id_producto", how="left")


def generar_lista_compra_v8(pred: pd.DataFrame) -> pd.DataFrame:
    """Lista de compra: solo el déficit entre necesidad predicha y stock actual."""
    df = pred.copy()
    df["cantidad_a_comprar"] = (
        df["pred_cantidad_siguiente_semana"] - df["stock_inicio_semana"]
    ).clip(lower=0).round(0).astype(int)
    df = df[df["cantidad_a_comprar"] > 0].copy()
    if df.empty:
        return df
    orden_riesgo = {"crítico": 0, "alto": 1, "medio": 2, "bajo": 3}
    df["_orden"] = df["nivel_riesgo"].map(orden_riesgo).fillna(4)
    df["costo_estimado_compra"] = (df["cantidad_a_comprar"] * df["costo_unitario_promedio_reciente"]).round(2)
    return df.sort_values(["_orden", "costo_estimado_compra"], ascending=[True, False]).drop(columns="_orden")


# ══════════════════════════════════════════════════════════════════════════
# Predicción de prioridad de albergue (adaptado de src/predict.py del paquete v8)
# ══════════════════════════════════════════════════════════════════════════
def recomendar_accion_albergue(prioridad: str) -> str:
    p = str(prioridad).lower().replace("í", "i")
    if p == "urgente":
        return "Asignar cupo si existe disponibilidad y activar validación clínica/social inmediata."
    if p == "alta":
        return "Priorizar en lista de espera y evaluar cupo durante la revisión diaria."
    if p == "media":
        return "Mantener seguimiento, confirmar documentos y reevaluar si cambia el tratamiento o alojamiento."
    return "No priorizar cupo por ahora; ofrecer orientación y seguimiento regular."


def explicar_prioridad_albergue(row: pd.Series | dict) -> list[str]:
    r = dict(row)
    motivos: list[str] = []
    if r.get("condicion_alojamiento") == "sin_alojamiento_seguro":
        motivos.append("No cuenta con alojamiento seguro en Lima.")
    elif r.get("condicion_alojamiento") == "temporal":
        motivos.append("Tiene alojamiento temporal o inestable.")

    distancia = float(r.get("distancia_lima_km", 0) or 0)
    if distancia >= 1000:
        motivos.append("Proviene de una zona muy alejada de Lima.")
    elif distancia >= 250:
        motivos.append("Proviene de provincia y requiere permanencia logística cerca del tratamiento.")

    fase = str(r.get("fase_tratamiento_actual", ""))
    if fase in {"tratamiento_intensivo", "diagnostico_inicial", "reingreso", "post_hospitalizacion"}:
        motivos.append("La fase actual exige seguimiento cercano o continuidad del tratamiento.")

    if r.get("riesgo_clinico_reciente") == "alerta_medica":
        motivos.append("Hay una alerta clínica reciente que debe validar el equipo responsable.")
    elif r.get("riesgo_clinico_reciente") == "seguimiento_cercano":
        motivos.append("Presenta necesidad de seguimiento cercano por evolución reciente.")

    if r.get("estado_nutricional") in {"desnutricion_severa", "desnutricion_moderada"}:
        motivos.append("El estado nutricional incrementa la vulnerabilidad del caso.")

    if r.get("apoyo_familiar") in {"sin_red_clara", "bajo"}:
        motivos.append("La red de apoyo familiar es limitada.")

    if r.get("situacion_economica") == "muy_vulnerable":
        motivos.append("La situación económica es muy vulnerable.")
    elif r.get("situacion_economica") == "vulnerable":
        motivos.append("La situación económica requiere apoyo social.")

    if r.get("continuidad_tratamiento") == "interrumpida":
        motivos.append("Existe riesgo de interrupción del tratamiento o controles.")
    elif r.get("continuidad_tratamiento") == "en_riesgo":
        motivos.append("La continuidad del tratamiento está en riesgo.")

    edad = int(r.get("edad_anios", 99) or 99)
    if edad <= 5:
        motivos.append("La edad temprana aumenta la necesidad de acompañamiento y estabilidad.")

    return motivos[:5] or ["No se detectaron factores críticos fuertes con los datos ingresados."]


def predict_patient_priority_v8(df_features: pd.DataFrame) -> dict[str, Any]:
    bundle = _load_model(MODEL_PRIORIDAD_PATH)
    features = bundle["features"]
    model = bundle["model"]
    labels = bundle["labels"]
    data = df_features.copy()
    for col in bundle.get("categorical_features", []):
        if col in data.columns:
            data[col] = data[col].astype(str)
    X = data[features]

    pred_code = int(model.predict(X)[0])
    encoder_classes = bundle.get("encoder_classes", labels)
    prioridad = str(encoder_classes[pred_code]) if 0 <= pred_code < len(encoder_classes) else str(pred_code)
    probas = model.predict_proba(X)[0]
    proba_dict_raw = {str(encoder_classes[i]): float(round(probas[i], 4)) for i in range(len(encoder_classes))}
    proba_dict = {label: proba_dict_raw.get(label, 0.0) for label in labels}
    score_map = bundle.get("score_map", {"baja": 25.0, "media": 53.0, "alta": 71.0, "urgente": 90.0})
    score = float(sum(proba_dict.get(label, 0.0) * float(score_map.get(label, 50.0)) for label in labels))

    return {
        "prioridad": prioridad,
        "puntaje_prioridad": round(score, 1),
        "probabilidades": proba_dict,
        "accion_sugerida": recomendar_accion_albergue(prioridad),
        "motivos": explicar_prioridad_albergue(data.iloc[0].to_dict()),
    }
