"""
ALDIMI Predict — Dashboard 100% componentes nativos de Streamlit.

El tema visual (colores, fondo, tipografía) se define en `.streamlit/config.toml`;
este archivo no contiene CSS ni HTML manual.

Actualización (6 jul, v10): tras revisarlo con el profesor, el equipo decidió
mantener el modelo ANTERIOR de pacientes (el binario `risk_model_rf_binary.joblib`,
que ya tenía el visto bueno) y usar del paquete del compañero (Aldani2.rar)
SOLO el modelo nuevo de inventario:
  - Productos: ExtraTreesRegressor que predice la cantidad necesaria de cada
    producto la próxima semana (catálogo nutricional de 36 productos). El
    costo se calcula después (cantidad_predicha × costo unitario promedio
    reciente) y se generan nivel de riesgo, días de cobertura y una
    recomendación de compra en texto plano. Vive en `aldani2_lib.py`.
  - Pacientes: el modelo nuevo de prioridad (XGBoost, 4 clases) que trajo el
    paquete del compañero NO se usa en este dashboard — el profesor ya había
    aprobado el modelo binario anterior y el equipo prefirió no reemplazarlo.
    Queda disponible en `models/modelo_prioridad_albergue.pkl` /
    `aldani2_lib.predict_patient_priority_v8` por si se retoma más adelante.
    La vista de pacientes sigue el modelo binario de siempre (probabilidad de
    "Alto" riesgo → banda Alta / Media / Baja).
Se mantiene retirada la vista "Modo Desarrollador": el dashboard tiene una
sola vista operativa.

El concepto de "familias" se mantiene igual que antes para el inventario: se
deriva del número de pacientes activos registrados y escala la predicción de
productos (factor_familias = familias / 50, ya que el histórico de
entrenamiento representa una ocupación base de 50 familias).

Estructura
──────────
 1. Configuración de página
 2. Carga de datos de pacientes (CSV)              → cargar_datos() / cargar_pacientes_actual()
 3. Alta / edición de pacientes                     → guardar_paciente()
 4. Prioridad de pacientes (modelo binario anterior) → evaluar_prioridad() / evaluar_todos_los_pacientes()
 5. VISTA OPERATIVA (única vista)                   → vista_usuario()
 6. Enrutado principal                              → main()

El catálogo y las predicciones de productos nuevos viven en aldani2_lib.py
(adaptación a CSV de src/predict.py y src/features.py del paquete que
compartió el compañero, sin la base SQLite que traía originalmente).
"""

import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# Compatibilidad con modelos serializados en una versión distinta de scikit-learn
# (necesario tanto para el modelo binario de pacientes como para los modelos
# nuevos de inventario que carga aldani2_lib).
from sklearn.compose import _column_transformer as _ct_mod
if not hasattr(_ct_mod, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    _ct_mod._RemainderColsList = _RemainderColsList

try:
    BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE = os.getcwd()

if BASE not in sys.path:
    sys.path.insert(0, BASE)

import aldani2_lib as v9  # catálogo y modelo nuevo de productos (ver docstring)

# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN DE PÁGINA
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ALDIMI Predict",
    layout="wide",
    initial_sidebar_state="expanded",
)

PACIENTES_CSV = os.path.join(BASE, "data", "processed", "aldimi_pacientes_sintetico.csv")
PACIENTE_ID_COL = "id_paciente"
MODELO_PACIENTES_PATH = os.path.join(BASE, "models", "risk_model_rf_binary.joblib")


def money(v: float) -> str:
    try:
        return f"S/ {float(v):,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    except Exception:
        return "S/ 0,00"


# ══════════════════════════════════════════════════════════════════════════════
# 2. CARGA DE DATOS DE PACIENTES
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def cargar_datos() -> pd.DataFrame:
    return pd.read_csv(PACIENTES_CSV)


def cargar_pacientes_actual() -> pd.DataFrame:
    """Relee el CSV de pacientes en caliente (para reflejar altas/ediciones recién guardadas)."""
    return pd.read_csv(PACIENTES_CSV)


# ══════════════════════════════════════════════════════════════════════════════
# 3. ALTA / EDICIÓN DE PACIENTES
#    Persiste directamente sobre el CSV de pacientes. Conserva todas las
#    columnas históricas; solo agrega/actualiza las columnas que usa el
#    formulario actual.
# ══════════════════════════════════════════════════════════════════════════════
def guardar_paciente(datos: dict, id_existente=None) -> str:
    df_pac = cargar_pacientes_actual()
    if id_existente and id_existente in df_pac[PACIENTE_ID_COL].values:
        idx = df_pac.index[df_pac[PACIENTE_ID_COL] == id_existente][0]
        for k, v in datos.items():
            df_pac.loc[idx, k] = v
        nuevo_id = id_existente
    else:
        n = len(df_pac) + 1
        nuevo_id = f"PAC-NEW-{n:03d}"
        fila = {**datos, PACIENTE_ID_COL: nuevo_id}
        df_pac = pd.concat([df_pac, pd.DataFrame([fila])], ignore_index=True)
    df_pac.to_csv(PACIENTES_CSV, index=False)
    st.cache_data.clear()
    return nuevo_id


# ══════════════════════════════════════════════════════════════════════════════
# 4. PRIORIDAD DE PACIENTES — modelo binario anterior (ya aprobado por el profesor)
#    Clasificador RandomForest binario: probabilidad de "Alto" riesgo → banda
#    Alta / Media / Baja según dos umbrales. Mismo modelo y mismos umbrales
#    que ya se venían usando antes de este cambio.
# ══════════════════════════════════════════════════════════════════════════════
BAJO_MAX = 0.35
ALTO_MIN = 0.67

# Valores neutros para completar variables del modelo que falten en un registro
# (defensivo: algún registro puede tener campos vacíos).
DEFAULTS_PACIENTE_MODELO = {
    "apoyo_familiar": "Moderado", "adherencia_tratamiento": "Alta", "acceso_medicamentos": "Completo",
    "estado_emocional_paciente": "Estable", "perdida_peso_reciente": 0, "num_comorbilidades": 0,
    "frecuencia_hospitalizacion_3m": 0,
}


@st.cache_resource
def cargar_modelo_pacientes():
    if not os.path.exists(MODELO_PACIENTES_PATH):
        return None
    return joblib.load(MODELO_PACIENTES_PATH)


def columnas_paciente_modelo(modelo, df_fila: pd.DataFrame) -> list:
    cols = [c for c in getattr(modelo, "feature_names_in_", []) if c in df_fila.columns]
    return cols or [c for c in df_fila.columns if c not in {PACIENTE_ID_COL, "nivel_riesgo"}]


def score_paciente_alto(modelo, fila: pd.DataFrame) -> float:
    """Probabilidad (0-1) de que el modelo binario clasifique al paciente como 'Alto'."""
    cols = columnas_paciente_modelo(modelo, fila)
    entrada = fila[cols].copy()
    for c, v in DEFAULTS_PACIENTE_MODELO.items():
        if c in entrada.columns:
            entrada[c] = entrada[c].fillna(v)
    probs = modelo.predict_proba(entrada)
    clases = list(getattr(modelo, "classes_", []))
    idx_alto = clases.index(1) if 1 in clases else (len(clases) - 1)
    return float(probs[0][idx_alto])


def banda_desde_score(score: float) -> str:
    if score < BAJO_MAX:
        return "baja"
    if score <= ALTO_MIN:
        return "media"
    return "alta"


def clasificador_paciente(modelo, fila: pd.DataFrame) -> dict:
    if modelo is None:
        return {"prioridad": "media", "puntaje_prioridad": 50.0}
    score = score_paciente_alto(modelo, fila)
    return {"prioridad": banda_desde_score(score), "puntaje_prioridad": round(score * 100, 1)}


def evaluar_prioridad(datos: dict) -> dict:
    modelo = cargar_modelo_pacientes()
    return clasificador_paciente(modelo, pd.DataFrame([datos]))


@st.cache_data
def evaluar_todos_los_pacientes(df_pac: pd.DataFrame) -> pd.DataFrame:
    """Evalúa la prioridad de todos los pacientes de una vez (para listas/resúmenes)."""
    modelo = cargar_modelo_pacientes()
    out = df_pac.copy()
    prioridades, puntajes = [], []
    for _, row in out.iterrows():
        res = clasificador_paciente(modelo, pd.DataFrame([row.to_dict()]))
        prioridades.append(res["prioridad"])
        puntajes.append(res["puntaje_prioridad"])
    out["prioridad"] = prioridades
    out["puntaje_prioridad"] = puntajes
    return out


def descripcion_prioridad(prioridad: str) -> str:
    return {
        "alta": "🔴 Necesita atención pronto",
        "media": "🟡 Revisar cuando se pueda",
        "baja": "🟢 Sin urgencia por ahora",
    }.get(str(prioridad).lower(), str(prioridad))


def señales_paciente(p) -> str:
    señales = []
    if p.get("etapa_cancer") in ("III", "IV"):
        señales.append(f"etapa {p.get('etapa_cancer')}")
    if p.get("presencia_infeccion") in (1, 1.0, True):
        señales.append("infección activa")
    nutri = p.get("estado_nutricional")
    if nutri not in ("Normal", None) and pd.notna(nutri):
        señales.append("desnutrición")
    if p.get("perdida_peso_reciente") in (1, 1.0, True):
        señales.append("pérdida de peso reciente")
    if p.get("apoyo_familiar") == "Limitado":
        señales.append("poco apoyo familiar")
    return ", ".join(señales[:3]) if señales else "sin señales de alarma"


# Opciones de los campos del formulario (mismas categorías que vio el modelo
# durante el entrenamiento — ver feature_names_in_ / categories_ del pipeline).
OPCIONES_SEXO = ["Femenino", "Masculino"]
OPCIONES_LUGAR = ["Sierra sur", "Sierra norte", "Sierra centro", "Selva", "Costa norte", "Costa sur", "Lima"]
OPCIONES_DIAGNOSTICO = [
    "Leucemia linfoblástica aguda", "Leucemia mieloide aguda", "Linfoma de Hodgkin",
    "Linfoma no Hodgkin", "Tumor cerebral", "Neuroblastoma", "Tumor de Wilms",
    "Osteosarcoma", "Retinoblastoma", "Rabdomiosarcoma",
]
OPCIONES_ETAPA = ["I", "II", "III", "IV"]
OPCIONES_APOYO = ["Limitado", "Moderado", "Fuerte"]
OPCIONES_ADHERENCIA = ["Baja", "Media", "Alta"]
OPCIONES_ACCESO = ["Limitado", "Parcial", "Completo"]
OPCIONES_NUTRICION = ["Normal", "Desnutrición leve", "Desnutrición severa"]
OPCIONES_EMOCIONAL = ["Ansioso", "Deprimido", "Estable"]


# ══════════════════════════════════════════════════════════════════════════════
# 5. VISTA OPERATIVA — Personal del albergue (única vista del dashboard)
# ══════════════════════════════════════════════════════════════════════════════
def vista_usuario(ctx: dict) -> None:
    df_pac = ctx["df_pac"]
    df_pac_eval = evaluar_todos_los_pacientes(df_pac)
    pac_alta = df_pac_eval[df_pac_eval["prioridad"] == "alta"].sort_values(
        "puntaje_prioridad", ascending=False)

    familias = ctx["familias_actuales"]
    factor_familias = familias / v9.FAMILIAS_BASE_ENTRENAMIENTO
    pred_productos = calcular_predicciones_cacheado(factor_familias)
    en_riesgo = pred_productos[pred_productos["nivel_riesgo"].isin(["crítico", "alto"])]
    sin_stock = pred_productos[pred_productos["stock_inicio_semana"] <= 0]
    presupuesto_semana = float(pred_productos["pred_costo_siguiente_semana"].sum())

    st.title("Panel de gestión del albergue")
    st.caption(f"{familias} familias alojadas (según pacientes activos registrados) · "
               "Lo importante primero: qué reponer y a quién atender.")

    tab_resumen, tab_inv, tab_pac = st.tabs(["Resumen Operativo", "Inventario", "Pacientes"])

    # ── RESUMEN OPERATIVO ─────────────────────────────────────────────────────
    with tab_resumen:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Productos sin stock", len(sin_stock),
                  delta="Reponer de inmediato" if len(sin_stock) else "Sin pendientes",
                  delta_color="inverse" if len(sin_stock) else "normal")
        c2.metric("Productos en riesgo", len(en_riesgo),
                  delta="Cobertura corta" if len(en_riesgo) else "Sin pendientes",
                  delta_color="inverse" if len(en_riesgo) else "normal")
        c3.metric("Presupuesto próxima semana", money(presupuesto_semana),
                  delta=f"Para {familias} familias", delta_color="off")
        c4.metric("Pacientes en prioridad alta", len(pac_alta),
                  delta="Atender primero" if len(pac_alta) else "Sin casos",
                  delta_color="inverse" if len(pac_alta) else "normal")

        st.divider()
        col_a, col_b = st.columns(2, gap="medium")

        with col_a:
            with st.container(border=True):
                st.subheader("Reposición urgente — esta semana")
                urgentes_prod = pred_productos[pred_productos["nivel_riesgo"].isin(["crítico", "alto"])].sort_values(
                    "dias_cobertura")
                if urgentes_prod.empty:
                    st.success("Sin productos en riesgo esta semana.")
                else:
                    for _, r in urgentes_prod.head(6).iterrows():
                        icono = "🔴" if r["nivel_riesgo"] == "crítico" else "🟠"
                        st.error(f"{icono} **{r['nombre_producto']}** · {r['nombre_categoria']} — "
                                 f"{r['dias_cobertura']:.1f} días de cobertura · {r['recomendacion']}")
                    if len(urgentes_prod) > 6:
                        st.caption(f"… y {len(urgentes_prod) - 6} más en la pestaña Inventario.")

        with col_b:
            with st.container(border=True):
                st.subheader("Pacientes con prioridad alta")
                if pac_alta.empty:
                    st.success("Sin pacientes en prioridad alta ahora mismo.")
                else:
                    for _, p in pac_alta.head(6).iterrows():
                        st.error(f"🔴 **{p['id_paciente']}** · {int(p['edad'])} años · "
                                 f"{p.get('diagnostico', 'sin diagnóstico registrado')} — "
                                 f"señales: {señales_paciente(p)}")
                    if len(pac_alta) > 6:
                        st.caption(f"… y {len(pac_alta) - 6} más en la pestaña Pacientes.")

    # ── INVENTARIO ────────────────────────────────────────────────────────────
    with tab_inv:
        st.header("Plan de reposición — próxima semana")

        df_pac_fresco = cargar_pacientes_actual()
        familias_sugeridas = max(1, len(df_pac_fresco))
        familias_input = st.number_input(
            "¿Cuántas familias hay hoy en el albergue?",
            min_value=1, max_value=200, value=familias_sugeridas, step=1,
            help="Se sugiere según el número de pacientes activos registrados; ajústalo si hace "
                 "falta. La cantidad y el costo estimados de abajo se recalculan con este número.")
        if "familiares_presentes" in df_pac_fresco.columns and len(df_pac_fresco):
            personas_totales = len(df_pac_fresco) + int(df_pac_fresco["familiares_presentes"].fillna(0).sum())
            st.caption(f"{familias_sugeridas} familias (una por paciente activo) · "
                       f"{personas_totales} personas en total, contando acompañantes.")

        factor_input = familias_input / v9.FAMILIAS_BASE_ENTRENAMIENTO
        pred = calcular_predicciones_cacheado(factor_input)

        f1, f2, f3 = st.columns([2, 2, 1], gap="medium")
        with f1:
            filtro = st.radio("Mostrar", ["Solo con acción pendiente", "Todos los productos"],
                              horizontal=True, label_visibility="collapsed")
        with f2:
            busqueda = st.text_input("Buscar", placeholder="Buscar por nombre…",
                                     label_visibility="collapsed")

        vista = pred if filtro == "Todos los productos" else pred[pred["nivel_riesgo"] != "bajo"]
        if busqueda.strip():
            q = busqueda.strip().lower()
            vista = vista[vista["nombre_producto"].str.lower().str.contains(q, regex=False)]

        tabla = vista.copy()
        tabla["Situación"] = tabla.apply(
            lambda r: f"{'🔴' if r['nivel_riesgo']=='crítico' else '🟠' if r['nivel_riesgo']=='alto' else '🟡' if r['nivel_riesgo']=='medio' else '🟢'} "
                      f"{r['dias_cobertura']:.0f} días de cobertura", axis=1)
        tabla = tabla.rename(columns={
            "nombre_producto": "Producto", "nombre_categoria": "Categoría",
            "stock_inicio_semana": "Stock actual",
            "pred_cantidad_siguiente_semana": "Cantidad estimada/semana",
            "pred_costo_siguiente_semana": "Costo estimado", "recomendacion": "Recomendación",
        })
        tabla["Costo estimado"] = tabla["Costo estimado"].apply(money)

        with f3:
            st.download_button("Descargar lista (CSV)",
                               vista.rename(columns={
                                   "nombre_producto": "producto", "nombre_categoria": "categoria",
                                   "stock_inicio_semana": "stock_actual",
                                   "pred_cantidad_siguiente_semana": "cantidad_estimada_semana",
                                   "pred_costo_siguiente_semana": "costo_estimado",
                               }).to_csv(index=False).encode("utf-8-sig"),
                               file_name="plan_reposicion_aldimi.csv",
                               mime="text/csv", use_container_width=True)

        if tabla.empty:
            st.success("Todo el almacén está cubierto para la próxima semana."
                       if not busqueda.strip() else "Sin resultados para esa búsqueda.")
        else:
            st.dataframe(
                tabla[["Producto", "Categoría", "Stock actual", "Cantidad estimada/semana",
                       "Costo estimado", "Situación", "Recomendación"]],
                use_container_width=True, hide_index=True, height=340,
                column_config={"Producto": st.column_config.TextColumn("Producto", width="large")})
            st.caption(f"{len(sin_stock)} sin stock · {len(en_riesgo)} en riesgo (crítico o alto) · "
                       f"calculado con **{familias_input} familias** · "
                       f"presupuesto estimado **{money(pred['pred_costo_siguiente_semana'].sum())}**.")

        st.divider()
        st.subheader("Registrar llegada de productos")
        catalogo = v9.cargar_catalogo_v8()
        with st.form("form_ingreso_v9"):
            opciones = dict(zip(catalogo["nombre_producto"], catalogo["id_producto"]))
            i1, i2, i3 = st.columns([3, 1, 2])
            with i1:
                producto_sel = st.selectbox("Producto", list(opciones.keys()))
            with i2:
                cantidad_ingreso = st.number_input("Cantidad recibida", min_value=0.0, step=1.0)
            with i3:
                nota_ingreso = st.text_input("Nota (opcional)", placeholder="ej. donación, compra…")
            enviado_ingreso = st.form_submit_button("Guardar ingreso", use_container_width=True)

        if enviado_ingreso:
            if cantidad_ingreso <= 0:
                st.warning("Ingresa una cantidad mayor a 0 antes de guardar.")
            else:
                v9.guardar_ingreso_v8(int(opciones[producto_sel]), cantidad_ingreso, nota_ingreso)
                st.cache_data.clear()
                st.success(f"Registrado: +{cantidad_ingreso:g} de **{producto_sel}**. "
                           "Cambia el número de familias o vuelve a esta pestaña para ver la tabla actualizada.")

        with st.expander("Consultar un producto con otro número de familias"):
            st.caption("Para probar un escenario distinto sin cambiar el número de arriba.")
            c1, c2, c3 = st.columns([2, 1, 1], gap="medium")
            with c1:
                prod_consulta = st.selectbox("Producto", pred["nombre_producto"].tolist(), key="consulta_producto_v9")
            with c2:
                familias_hip = st.number_input("Familias a probar", 1, 200, familias_input, step=1,
                                               key="consulta_familias_v9")
            with c3:
                consultar = st.button("Consultar", use_container_width=True)
            if consultar:
                pred_hip = calcular_predicciones_cacheado(familias_hip / v9.FAMILIAS_BASE_ENTRENAMIENTO)
                fila = pred_hip[pred_hip["nombre_producto"] == prod_consulta].iloc[0]
                st.info(f"Con **{familias_hip} familias**: {fila['dias_cobertura']:.1f} días de cobertura "
                        f"({fila['nivel_riesgo']}) · cantidad estimada: {fila['pred_cantidad_siguiente_semana']:g} · "
                        f"costo estimado: {money(fila['pred_costo_siguiente_semana'])}. {fila['recomendacion']}")

        with st.expander("Ver movimientos de inventario registrados a mano"):
            df_mov_ver = v9.cargar_movimientos_v8()
            if df_mov_ver.empty:
                st.caption("Todavía no se registró ningún ingreso manual.")
            else:
                catalogo_map = dict(zip(catalogo["id_producto"], catalogo["nombre_producto"]))
                df_mov_ver = df_mov_ver.copy()
                df_mov_ver["producto"] = df_mov_ver["id_producto"].map(catalogo_map)
                st.dataframe(df_mov_ver.sort_values("fecha", ascending=False)[["fecha", "producto", "cantidad", "nota"]],
                            use_container_width=True, hide_index=True)

    # ── PACIENTES ─────────────────────────────────────────────────────────────
    with tab_pac:
        st.header("Pacientes que requieren atención primero")
        if pac_alta.empty:
            st.success("Ningún paciente en prioridad alta ahora mismo.")
        else:
            tabla_pac = pac_alta[["id_paciente", "edad", "diagnostico", "lugar_procedencia",
                                   "etapa_cancer", "prioridad"]].copy()
            tabla_pac["prioridad"] = tabla_pac["prioridad"].apply(descripcion_prioridad)
            tabla_pac.columns = ["Paciente", "Edad", "Diagnóstico", "Procedencia", "Etapa", "Prioridad"]
            st.dataframe(tabla_pac, use_container_width=True, hide_index=True, height=280)
            st.caption(f"{len(pac_alta)} de {len(df_pac_eval)} pacientes en prioridad alta. "
                       "Confirmar cada caso con el equipo médico/social.")

        st.divider()
        st.subheader("Situación general de los pacientes")
        for banda, etiqueta in [("alta", "Prioridad alta"), ("media", "En revisión"), ("baja", "Prioridad baja")]:
            cnt = int((df_pac_eval["prioridad"] == banda).sum())
            pct = cnt / len(df_pac_eval) if len(df_pac_eval) else 0
            st.progress(pct, text=f"{etiqueta}: {cnt} pacientes ({pct*100:.0f}%)")

        st.divider()
        accion_paciente = st.radio(
            "¿Qué quieres hacer?", ["➕ Registrar nuevo paciente", "✏️ Editar paciente existente"],
            horizontal=True, label_visibility="collapsed")
        if accion_paciente == "➕ Registrar nuevo paciente":
            formulario_paciente(ctx)
        else:
            editar_paciente_existente(ctx)


@st.cache_data
def calcular_predicciones_cacheado(factor_familias: float) -> pd.DataFrame:
    return v9.calcular_predicciones_productos_v8(factor_familias)


def formulario_paciente(ctx: dict) -> None:
    """Registra un paciente nuevo y evalúa su prioridad con el modelo aprobado.
    Se piden los datos básicos más las variables psicosociales que usa el
    modelo; el resto de columnas clínicas que también usa el modelo se
    completan con valores neutros por defecto (igual que en el prototipo ya
    validado con el equipo)."""
    with st.form("form_paciente_v10"):
        g1, g2, g3 = st.columns(3)
        with g1:
            edad = st.number_input("Edad (años)", 0, 18, 7)
            sexo = st.selectbox("Sexo", OPCIONES_SEXO)
            diagnostico = st.selectbox("Diagnóstico", OPCIONES_DIAGNOSTICO)
            etapa = st.selectbox("Etapa del cáncer", OPCIONES_ETAPA)
        with g2:
            lugar = st.selectbox("Procedencia", OPCIONES_LUGAR)
            distancia = st.number_input("Distancia a Lima (km)", 10, 1400, 600)
            familiares_presentes = st.number_input(
                "Familiares que lo acompañan", 0, 10, 1,
                help="Cuántos familiares están presentes en el albergue junto al paciente. "
                     "Es solo informativo: no participa en la evaluación de prioridad.")
            apoyo = st.selectbox("Apoyo familiar", OPCIONES_APOYO, index=OPCIONES_APOYO.index("Moderado"))
        with g3:
            adherencia = st.selectbox("¿Sigue el tratamiento?", OPCIONES_ADHERENCIA, index=OPCIONES_ADHERENCIA.index("Alta"))
            acceso = st.selectbox("Acceso a medicamentos", OPCIONES_ACCESO, index=OPCIONES_ACCESO.index("Completo"))
            nutricion = st.selectbox("Estado nutricional", OPCIONES_NUTRICION, index=0)
            emocional = st.selectbox("Estado emocional", OPCIONES_EMOCIONAL, index=OPCIONES_EMOCIONAL.index("Estable"))

        registrar = st.checkbox("Registrar este paciente como nuevo ingreso al guardar")
        enviado = st.form_submit_button("Evaluar prioridad de atención", use_container_width=True)

    if not enviado:
        return

    datos = {
        "edad": edad, "sexo": sexo, "lugar_procedencia": lugar, "distancia_origen_km": distancia,
        "grado_instruccion_cuidador": "Secundaria", "diagnostico": diagnostico, "etapa_cancer": etapa,
        "tipo_tratamiento": "Quimioterapia", "meses_en_tratamiento": 1, "num_reingresos": 0,
        "motivo_ingreso": "Tratamiento", "motivo_reingreso": "Primer ingreso",
        "estado_fisico": "Caminando", "presencia_infeccion": 0,
        "frecuencia_hospitalizacion_3m": 0, "perdida_peso_reciente": 0, "num_comorbilidades": 0,
        "apoyo_familiar": apoyo, "adherencia_tratamiento": adherencia,
        "acceso_medicamentos": acceso, "estado_nutricional": nutricion,
        "estado_emocional_paciente": emocional, "nivel_riesgo": "Medio",
    }
    resultado = evaluar_prioridad(datos)
    prioridad = resultado["prioridad"]

    if prioridad == "alta":
        st.error(f"**{descripcion_prioridad(prioridad)}**")
    elif prioridad == "media":
        st.warning(f"**{descripcion_prioridad(prioridad)}**")
    else:
        st.success(f"**{descripcion_prioridad(prioridad)}**")
    st.caption(f"Señales a revisar: {señales_paciente(datos)}.")
    st.caption("Resultado orientativo; la evaluación clínica y social la confirma el equipo responsable.")

    if registrar:
        datos_guardar = dict(datos)
        datos_guardar["familiares_presentes"] = familiares_presentes
        nuevo_id = guardar_paciente(datos_guardar)
        st.success(f"Registrado como nuevo paciente: **{nuevo_id}**. Aparecerá en las listas de arriba "
                   "la próxima vez que se actualice esta pantalla.")


def editar_paciente_existente(ctx: dict) -> None:
    """Carga un paciente ya registrado, permite ajustar las variables
    psicosociales que usa el modelo de prioridad y guarda los cambios. Los
    datos clínicos/demográficos de base quedan como referencia (no editables
    aquí), igual que antes."""
    df_pac_actual = cargar_pacientes_actual()
    if df_pac_actual.empty:
        st.info("No hay pacientes registrados todavía.")
        return

    id_sel = st.selectbox("Selecciona un paciente", df_pac_actual[PACIENTE_ID_COL].tolist(),
                          key="sel_editar_paciente_v10")
    base = df_pac_actual[df_pac_actual[PACIENTE_ID_COL] == id_sel].iloc[0]
    banda_antes = evaluar_prioridad(base.to_dict())["prioridad"]

    st.caption(f"Datos de referencia (no editables aquí): {base.get('edad', '?')} años · "
               f"{base.get('diagnostico', 'sin diagnóstico registrado')} · "
               f"etapa {base.get('etapa_cancer', '?')} · "
               f"{base.get('meses_en_tratamiento', '?')} meses en tratamiento.")

    def idx(opciones, valor, default=0):
        return opciones.index(valor) if valor in opciones else default

    with st.form(f"form_editar_v10_{id_sel}"):
        st.caption("Variables que el albergue sí puede gestionar:")
        e1, e2, e3 = st.columns(3)
        with e1:
            apoyo = st.selectbox("Apoyo familiar", OPCIONES_APOYO,
                                 index=idx(OPCIONES_APOYO, base.get("apoyo_familiar"), OPCIONES_APOYO.index("Moderado")))
            adherencia = st.selectbox("¿Sigue el tratamiento?", OPCIONES_ADHERENCIA,
                                      index=idx(OPCIONES_ADHERENCIA, base.get("adherencia_tratamiento"), OPCIONES_ADHERENCIA.index("Alta")))
        with e2:
            acceso = st.selectbox("Acceso a medicamentos", OPCIONES_ACCESO,
                                  index=idx(OPCIONES_ACCESO, base.get("acceso_medicamentos"), OPCIONES_ACCESO.index("Completo")))
            nutricion = st.selectbox("Estado nutricional", OPCIONES_NUTRICION,
                                     index=idx(OPCIONES_NUTRICION, base.get("estado_nutricional"), 0))
        with e3:
            emocional = st.selectbox("Estado emocional", OPCIONES_EMOCIONAL,
                                     index=idx(OPCIONES_EMOCIONAL, base.get("estado_emocional_paciente"), OPCIONES_EMOCIONAL.index("Estable")))
            familiares_presentes = st.number_input(
                "Familiares que lo acompañan", 0, 10,
                int(base["familiares_presentes"]) if "familiares_presentes" in base.index
                and pd.notna(base["familiares_presentes"]) else 1)
        guardar_btn = st.form_submit_button("Guardar cambios", use_container_width=True)

    if not guardar_btn:
        return

    datos = dict(base)
    datos.update({
        "apoyo_familiar": apoyo, "adherencia_tratamiento": adherencia,
        "acceso_medicamentos": acceso, "estado_nutricional": nutricion,
        "estado_emocional_paciente": emocional, "familiares_presentes": familiares_presentes,
    })
    datos.pop(PACIENTE_ID_COL, None)
    guardar_paciente(datos, id_existente=id_sel)

    banda_despues = evaluar_prioridad(datos)["prioridad"]
    if banda_antes != banda_despues:
        st.success(f"Guardado. Prioridad: {descripcion_prioridad(banda_antes)} → "
                   f"**{descripcion_prioridad(banda_despues)}**")
    else:
        st.success(f"Guardado. Prioridad: {descripcion_prioridad(banda_despues)}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. ENRUTADO PRINCIPAL — una sola vista (se mantiene retirado Modo Desarrollador)
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    try:
        df_pac = cargar_datos()
    except FileNotFoundError as e:
        st.error(f"Archivos no encontrados: {e}")
        st.stop()

    ctx = {
        "df_pac": df_pac,
        "familias_actuales": max(1, len(df_pac)),
    }

    with st.sidebar:
        st.title("ALDIMI Predict")
        st.caption("Albergue Divina Misericordia")
        st.divider()

        st.subheader("Resumen de hoy")
        factor = ctx["familias_actuales"] / v9.FAMILIAS_BASE_ENTRENAMIENTO
        pred = calcular_predicciones_cacheado(factor)
        df_pac_eval = evaluar_todos_los_pacientes(df_pac)
        en_riesgo = int(pred["nivel_riesgo"].isin(["crítico", "alto"]).sum())
        alta = int(df_pac_eval["prioridad"].eq("alta").sum())
        st.markdown(
            f"- **{en_riesgo}** productos en riesgo\n"
            f"- **{alta}** pacientes en prioridad alta\n"
            f"- **{ctx['familias_actuales']}** familias alojadas\n"
            f"- **{money(pred['pred_costo_siguiente_semana'].sum())}** presupuesto próxima semana"
        )
        st.divider()
        st.caption("Las recomendaciones apoyan la decisión del equipo; no reemplazan "
                   "el criterio del personal de salud.")
        st.caption("ML 1ACC0057 · UPC · Julio 2026")

    vista_usuario(ctx)


main()
