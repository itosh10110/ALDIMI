# -*- coding: utf-8 -*-
"""
ALDIMI — Panel Operativo (prototipo aparte)
════════════════════════════════════════════════════════════════════════════
Vista simplificada para el personal del albergue, SIN gráficos ni jerga
técnica — por pedido explícito del profesor Jairo (feedback del 6 jul,
transmitido por el equipo):

  1. "Si aumentan las familias, el consumo debe recalcularse automáticamente
     y mostrar mayor consumo / menos días de cobertura."
  2. "Cero tecnicismos: nada de scores, % de confianza, accuracy ni términos
     de ML en la vista principal. Debe ser operativo y directo
     (ej. 'Faltan 5 días para que se acabe el arroz')."
  3. Debe permitir tanto VER como INTRODUCIR datos (no solo lectura).

Este archivo es un PROTOTIPO APARTE: no modifica `dashboard.py` (el dashboard
completo, con vista técnica, que ya construyó el equipo). Se ejecuta solo:

    streamlit run dashboard_operativo.py

Cómo se calcula "cuántos días quedan" y por qué escala con las familias
------------------------------------------------------------------------
En vez de pedirle a un modelo de caja negra que "aprenda" el efecto de la
ocupación (se probó en notebooks/modelo_cantidad_inventario.ipynb, sección 7,
y el histórico solo cubre 40-60 familias — no es confiable extrapolar a 100),
se usa una tasa de consumo histórica **por familia** para cada producto:

    tasa_percapita = promedio histórico de (consumo_semanal / familias_en_esa_semana)
    consumo_esperado = tasa_percapita × familias_que_ingresa_el_personal
    días_de_cobertura = stock_actual / (consumo_esperado / 7)

Esto garantiza —por construcción, no por esperanza— que más familias implica
más consumo esperado y por lo tanto menos días de cobertura. Es simple,
verificable a mano y no requiere explicarle a nadie qué es un "score".

Persistencia de datos
-----------------------
Este prototipo lee y escribe directamente sobre los CSV de
`data/processed/` (no sobre `data/aldimi_core.db`): es más simple de
auditar para un equipo pequeño y no depende de que la BD SQLite esté
sincronizada. Los movimientos de inventario (ingresos registrados a mano)
se guardan en un archivo nuevo, `data/processed/movimientos_inventario.csv`,
para no tocar el histórico semanal que usan los notebooks de entrenamiento.
Si el equipo prefiere que todo pase por `aldimi_core.db`, es un cambio
puntual en las funciones `guardar_*` de más abajo.
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# Compatibilidad con modelos serializados en una versión distinta de scikit-learn
# (mismo parche que usa dashboard.py para poder cargar risk_model_rf_binary.joblib).
try:
    from sklearn.compose import _column_transformer as _ct_mod

    if not hasattr(_ct_mod, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        _ct_mod._RemainderColsList = _RemainderColsList
except Exception:
    pass


try:
    BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE = os.getcwd()

DATA = os.path.join(BASE, "data", "processed")
F_SEMANAL = os.path.join(DATA, "aldimi_dataset_semanal.csv")
F_CATALOGO = os.path.join(DATA, "catalogo_productos.csv")
F_PACIENTES = os.path.join(DATA, "aldimi_pacientes_sintetico.csv")
F_MOVIMIENTOS = os.path.join(DATA, "movimientos_inventario.csv")
F_MODELO_PACIENTES = os.path.join(BASE, "models", "risk_model_rf_binary.joblib")

COLS_MOVIMIENTOS = ["fecha", "codigo_articulo", "nombre_producto", "cantidad", "nota"]

# Variables del paciente que el personal del albergue SÍ puede intentar influir
# (psicosociales/operativas). Las clínicas (etapa, diagnóstico, etc.) quedan
# fijas: no tiene sentido "editarlas" para bajar el riesgo.
VARIABLES_ACCIONABLES = [
    "apoyo_familiar", "adherencia_tratamiento", "acceso_medicamentos",
    "estado_nutricional", "estado_emocional_paciente",
]

BANDA_A_TEXTO = {
    "Alto": "🔴 Necesita atención pronto",
    "Revisión": "🟡 Revisar cuando se pueda",
    "Bajo": "🟢 Sin urgencia por ahora",
}

BAJO_MAX = 0.35
ALTO_MIN = 0.67


# ══════════════════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS (sin dependencia de aldimi_core.db — solo CSV)
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data
def cargar_inventario():
    df = pd.read_csv(F_SEMANAL)
    cat = pd.read_csv(F_CATALOGO)
    return df, cat


def cargar_movimientos():
    """Ingresos registrados a mano por el personal (no está en el histórico)."""
    if os.path.exists(F_MOVIMIENTOS):
        return pd.read_csv(F_MOVIMIENTOS)
    return pd.DataFrame(columns=COLS_MOVIMIENTOS)


@st.cache_data
def cargar_pacientes():
    return pd.read_csv(F_PACIENTES)


@st.cache_resource
def cargar_modelo_pacientes():
    import joblib
    if not os.path.exists(F_MODELO_PACIENTES):
        return None
    return joblib.load(F_MODELO_PACIENTES)


# ══════════════════════════════════════════════════════════════════════════
# 2. MODELO DE CONSUMO "POR FAMILIA" (transparente, sin caja negra)
# ══════════════════════════════════════════════════════════════════════════
def calcular_tasas_percapita(df_inv: pd.DataFrame) -> pd.Series:
    """Consumo semanal promedio por familia, histórico, para cada producto real."""
    real = df_inv[df_inv["es_producto"] == True].copy()
    real["percapita"] = real["salidas_semana"] / real["ocupacion_albergue"].replace(0, np.nan)
    return real.groupby("codigo_articulo")["percapita"].mean().fillna(0.0)


def stock_actual(df_inv: pd.DataFrame, df_mov: pd.DataFrame) -> pd.Series:
    """Último stock del histórico + ingresos registrados a mano desde entonces."""
    real = df_inv[df_inv["es_producto"] == True].copy()
    ultimo = (real.sort_values("semana_del_año")
                  .groupby("codigo_articulo")["stock_fin_semana"].last()
                  .clip(lower=0))
    if len(df_mov):
        extra = df_mov.groupby("codigo_articulo")["cantidad"].sum()
        ultimo = ultimo.add(extra, fill_value=0)
    return ultimo


def dias_de_cobertura(stock: float, tasa_percapita: float, familias: float) -> float:
    """Días que alcanza el stock actual dado un número de familias. inf = no hay consumo."""
    if stock <= 0:
        return 0.0
    consumo_diario = (tasa_percapita * familias) / 7.0
    if consumo_diario <= 1e-9:
        return np.inf
    return stock / consumo_diario


def texto_situacion(dias: float) -> str:
    if dias <= 0:
        return "🔴 Ya no queda stock"
    if not np.isfinite(dias):
        return "⚪ Sin consumo reciente registrado"
    if dias <= 3:
        return f"🔴 ¡Urgente! Se acaba en {dias:.0f} día(s)"
    if dias <= 14:
        return f"🟠 Se acaba en {dias:.0f} días"
    return f"🟢 Alcanza para {dias:.0f}+ días"


def tabla_inventario(df_inv, cat, df_mov, familias: int) -> pd.DataFrame:
    tasas = calcular_tasas_percapita(df_inv)
    stock = stock_actual(df_inv, df_mov)
    cat_idx = cat.set_index("codigo_articulo")
    cat_idx = cat_idx[cat_idx["es_producto"] == True]

    filas = []
    for codigo in stock.index:
        if codigo not in cat_idx.index:
            continue
        info = cat_idx.loc[codigo]
        tasa = tasas.get(codigo, 0.0)
        st_actual = stock[codigo]
        dias = dias_de_cobertura(st_actual, tasa, familias)
        filas.append({
            "Código": codigo,
            "Producto": info["nombre_producto"],
            "Categoría": info["categoria_general"],
            "Unidad": info["unidad_medida"],
            "Stock actual": round(st_actual, 1),
            "Consumo esperado/semana": round(tasa * familias, 2),
            "_dias": dias,
            "Situación": texto_situacion(dias),
        })
    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return tabla
    return tabla.sort_values("_dias").drop(columns="_dias").reset_index(drop=True)


def guardar_ingreso(codigo: str, nombre: str, cantidad: float, nota: str) -> None:
    df_mov = cargar_movimientos()
    nueva = pd.DataFrame([{
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "codigo_articulo": codigo,
        "nombre_producto": nombre,
        "cantidad": cantidad,
        "nota": nota,
    }])
    df_mov = pd.concat([df_mov, nueva], ignore_index=True)
    df_mov.to_csv(F_MOVIMIENTOS, index=False)
    st.cache_data.clear()


# ══════════════════════════════════════════════════════════════════════════
# 3. PRIORIDAD DE PACIENTES (mismo modelo binario, cero números en pantalla)
# ══════════════════════════════════════════════════════════════════════════
def columnas_modelo(modelo, df_pac):
    cols = [c for c in getattr(modelo, "feature_names_in_", [])
            if c in df_pac.columns and c not in {"id_paciente", "nivel_riesgo"}]
    if cols:
        return cols
    return [c for c in df_pac.columns if c not in {"id_paciente", "nivel_riesgo"}]


def banda_prioridad(modelo, fila: pd.DataFrame) -> str:
    """Devuelve 'Alto' / 'Revisión' / 'Bajo' — el score interno nunca se muestra."""
    if modelo is None:
        return "Revisión"
    cols = columnas_modelo(modelo, fila)
    probs = modelo.predict_proba(fila[cols])
    clases = list(getattr(modelo, "classes_", []))
    idx_alto = clases.index(1) if 1 in clases else (clases.index("Alto") if "Alto" in clases else 1)
    score = probs[0][idx_alto]
    if score < BAJO_MAX:
        return "Bajo"
    if score <= ALTO_MIN:
        return "Revisión"
    return "Alto"


def señales_paciente(p: pd.Series) -> str:
    señales = []
    if p.get("etapa_cancer") in ("III", "IV"):
        señales.append(f"etapa {p['etapa_cancer']}")
    if p.get("presencia_infeccion") == 1:
        señales.append("infección activa")
    if p.get("estado_nutricional") not in ("Normal", None):
        señales.append("desnutrición")
    if p.get("perdida_peso_reciente") == 1:
        señales.append("pérdida de peso reciente")
    if p.get("apoyo_familiar") == "Limitado":
        señales.append("poco apoyo familiar")
    return ", ".join(señales[:3]) if señales else "sin señales de alarma"


def guardar_paciente(fila_dict: dict, id_existente: str | None = None) -> str:
    """Crea un paciente nuevo o actualiza uno existente. Devuelve el id usado."""
    df_pac = cargar_pacientes()
    if id_existente and id_existente in df_pac["id_paciente"].values:
        idx = df_pac.index[df_pac["id_paciente"] == id_existente][0]
        for k, v in fila_dict.items():
            df_pac.loc[idx, k] = v
        nuevo_id = id_existente
    else:
        n = len(df_pac) + 1
        nuevo_id = f"PAC-NEW-{n:03d}"
        fila_dict = {**fila_dict, "id_paciente": nuevo_id}
        df_pac = pd.concat([df_pac, pd.DataFrame([fila_dict])], ignore_index=True)
    df_pac.to_csv(F_PACIENTES, index=False)
    st.cache_data.clear()
    return nuevo_id


# ══════════════════════════════════════════════════════════════════════════
# 4. VISTA — ALMACÉN
# ══════════════════════════════════════════════════════════════════════════
def vista_almacen(df_inv, cat, familias: int) -> None:
    st.header("📦 Almacén")
    df_mov = cargar_movimientos()
    tabla = tabla_inventario(df_inv, cat, df_mov, familias)

    if tabla.empty:
        st.info("No hay productos para mostrar.")
        return

    urgentes = (tabla["Situación"].str.contains("Urgente|Ya no queda")).sum()
    if urgentes:
        st.error(f"{urgentes} producto(s) necesitan reposición urgente.")
    else:
        st.success("Ningún producto en estado urgente por ahora.")

    st.dataframe(tabla, use_container_width=True, hide_index=True, height=420)
    st.caption(f"Calculado con **{familias} familias**. Si cambia el número de "
               "familias alojadas (arriba, en la parte superior), esta tabla se "
               "recalcula sola.")

    st.divider()
    st.subheader("Registrar llegada de productos")
    with st.form("form_ingreso"):
        opciones = dict(zip(tabla["Producto"], tabla["Código"]))
        c1, c2, c3 = st.columns([3, 1, 2])
        with c1:
            producto_sel = st.selectbox("Producto", list(opciones.keys()))
        with c2:
            cantidad = st.number_input("Cantidad recibida", min_value=0.0, step=1.0)
        with c3:
            nota = st.text_input("Nota (opcional)", placeholder="ej. donación, compra…")
        enviado = st.form_submit_button("Guardar ingreso", use_container_width=True)

    if enviado:
        if cantidad <= 0:
            st.warning("Ingresa una cantidad mayor a 0 antes de guardar.")
        else:
            codigo = opciones[producto_sel]
            guardar_ingreso(codigo, producto_sel, cantidad, nota)
            st.success(f"Registrado: +{cantidad:g} de **{producto_sel}**. "
                       "La tabla de arriba ya lo refleja.")
            st.rerun()

    with st.expander("Ver movimientos registrados"):
        df_mov = cargar_movimientos()
        if df_mov.empty:
            st.caption("Todavía no se registró ningún ingreso manual.")
        else:
            st.dataframe(df_mov.sort_values("fecha", ascending=False),
                         use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
# 5. VISTA — PACIENTES
# ══════════════════════════════════════════════════════════════════════════
def vista_pacientes() -> None:
    st.header("🧒 Pacientes")
    df_pac = cargar_pacientes()
    modelo = cargar_modelo_pacientes()

    filas = []
    for _, p in df_pac.iterrows():
        banda = banda_prioridad(modelo, df_pac[df_pac["id_paciente"] == p["id_paciente"]])
        filas.append({
            "Paciente": p["id_paciente"], "Edad": p["edad"], "Diagnóstico": p["diagnostico"],
            "Prioridad": BANDA_A_TEXTO[banda], "_orden": {"Alto": 0, "Revisión": 1, "Bajo": 2}[banda],
            "Situación a revisar": señales_paciente(p),
        })
    tabla = pd.DataFrame(filas).sort_values("_orden").drop(columns="_orden")
    n_alto = (tabla["Prioridad"] == BANDA_A_TEXTO["Alto"]).sum()
    if n_alto:
        st.error(f"{n_alto} paciente(s) necesitan atención pronto.")
    else:
        st.success("Ningún paciente en prioridad alta ahora mismo.")
    st.dataframe(tabla, use_container_width=True, hide_index=True, height=380)

    st.divider()
    tab_nuevo, tab_editar = st.tabs(["➕ Registrar nuevo paciente", "✏️ Editar paciente existente"])

    with tab_nuevo:
        _formulario_paciente(modelo, df_pac, id_existente=None)

    with tab_editar:
        id_sel = st.selectbox("Selecciona un paciente", df_pac["id_paciente"].tolist(),
                              key="sel_editar")
        _formulario_paciente(modelo, df_pac, id_existente=id_sel)


def _formulario_paciente(modelo, df_pac: pd.DataFrame, id_existente) -> None:
    base = (df_pac[df_pac["id_paciente"] == id_existente].iloc[0]
            if id_existente else None)

    def val(campo, default):
        return base[campo] if base is not None else default

    with st.form(f"form_paciente_{id_existente or 'nuevo'}"):
        if base is not None:
            st.caption("Datos clínicos (de referencia, no editables aquí):")
            st.text(f"Edad: {base['edad']} · Diagnóstico: {base['diagnostico']} · "
                    f"Etapa: {base['etapa_cancer']} · Meses en tratamiento: {base['meses_en_tratamiento']}")

        st.caption("Variables que el albergue sí puede gestionar:")
        c1, c2, c3 = st.columns(3)
        with c1:
            apoyo = st.selectbox("Apoyo familiar", ["Fuerte", "Moderado", "Limitado"],
                                 index=["Fuerte", "Moderado", "Limitado"].index(val("apoyo_familiar", "Moderado")))
            adherencia = st.selectbox("¿Sigue el tratamiento?", ["Alta", "Media", "Baja"],
                                      index=["Alta", "Media", "Baja"].index(val("adherencia_tratamiento", "Alta")))
        with c2:
            acceso = st.selectbox("Acceso a medicamentos", ["Completo", "Parcial", "Limitado"],
                                  index=["Completo", "Parcial", "Limitado"].index(val("acceso_medicamentos", "Completo")))
            nutricion = st.selectbox("Estado nutricional", ["Normal", "Desnutrición leve", "Desnutrición severa"],
                                     index=["Normal", "Desnutrición leve", "Desnutrición severa"].index(val("estado_nutricional", "Normal")))
        with c3:
            emocional = st.selectbox("Estado emocional", ["Estable", "Ansioso", "Deprimido"],
                                     index=["Estable", "Ansioso", "Deprimido"].index(val("estado_emocional_paciente", "Estable")))

        if base is None:
            st.caption("Datos básicos del nuevo paciente:")
            d1, d2, d3 = st.columns(3)
            with d1:
                edad = st.number_input("Edad", 2, 17, 8)
                sexo = st.selectbox("Sexo", ["Masculino", "Femenino"])
            with d2:
                diagnostico = st.selectbox("Diagnóstico", [
                    "Leucemia linfoblástica aguda", "Leucemia mieloide aguda", "Linfoma de Hodgkin",
                    "Linfoma no Hodgkin", "Tumor cerebral", "Neuroblastoma", "Tumor de Wilms",
                    "Osteosarcoma", "Retinoblastoma", "Rabdomiosarcoma"])
                etapa = st.selectbox("Etapa del cáncer", ["I", "II", "III", "IV"])
            with d3:
                distancia = st.number_input("Distancia a Lima (km)", 10, 1400, 600)
                lugar = st.selectbox("Procedencia", ["Sierra sur", "Sierra norte", "Sierra centro",
                                                     "Selva", "Costa norte", "Costa sur", "Lima"])

        accion = "Guardar cambios" if base is not None else "Registrar paciente"
        enviado = st.form_submit_button(accion, use_container_width=True)

    if not enviado:
        return

    datos = dict(base) if base is not None else {
        "edad": edad, "sexo": sexo, "lugar_procedencia": lugar, "distancia_origen_km": distancia,
        "grado_instruccion_cuidador": "Secundaria", "diagnostico": diagnostico, "etapa_cancer": etapa,
        "tipo_tratamiento": "Quimioterapia", "meses_en_tratamiento": 1, "num_reingresos": 0,
        "motivo_ingreso": "Tratamiento", "motivo_reingreso": "Primer ingreso",
        "estado_fisico": "Caminando", "presencia_infeccion": 0,
        "frecuencia_hospitalizacion_3m": 0, "perdida_peso_reciente": 0, "num_comorbilidades": 0,
        "nivel_riesgo": "Medio",
    }
    datos.update({
        "apoyo_familiar": apoyo, "adherencia_tratamiento": adherencia,
        "acceso_medicamentos": acceso, "estado_nutricional": nutricion,
        "estado_emocional_paciente": emocional,
    })
    datos.pop("id_paciente", None)

    banda_previa = banda_prioridad(modelo, pd.DataFrame([dict(base)])) if base is not None else None
    banda_nueva = banda_prioridad(modelo, pd.DataFrame([datos]))

    nuevo_id = guardar_paciente(datos, id_existente=id_existente)

    if banda_previa and banda_previa != banda_nueva:
        st.success(f"Guardado ({nuevo_id}). Prioridad: {BANDA_A_TEXTO[banda_previa]} → "
                   f"**{BANDA_A_TEXTO[banda_nueva]}**")
    else:
        st.success(f"Guardado ({nuevo_id}). Prioridad: {BANDA_A_TEXTO[banda_nueva]}")
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# 6. ENRUTADO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    st.title("ALDIMI — Panel Operativo")
    st.caption("Vista simple para el día a día del albergue. Sin gráficos ni "
               "términos técnicos — solo lo que hay que saber y hacer hoy.")

    try:
        df_inv, cat = cargar_inventario()
    except FileNotFoundError as e:
        st.error(f"No se encontraron los datos: {e}")
        st.stop()

    ocupacion_ultima = int(df_inv.loc[df_inv["semana_del_año"].idxmax(), "ocupacion_albergue"])
    familias = st.number_input(
        "¿Cuántas familias hay hoy en el albergue?",
        min_value=1, max_value=200, value=ocupacion_ultima, step=1,
        help="Cambia este número y las tablas de abajo se recalculan solas.")

    tab_almacen, tab_pacientes = st.tabs(["📦 Almacén", "🧒 Pacientes"])
    with tab_almacen:
        vista_almacen(df_inv, cat, familias)
    with tab_pacientes:
        vista_pacientes()

    st.divider()
    st.caption("Prototipo aparte — no reemplaza el dashboard técnico del equipo "
               "(`dashboard.py`). Datos guardados directamente en los CSV de "
               "`data/processed/`.")


if __name__ == "__main__":
    main()
