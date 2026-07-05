"""
ALDIMI Predict — Dashboard 100% componentes nativos de Streamlit.

El tema visual (colores, fondo, tipografía) se define en `.streamlit/config.toml`;
este archivo no contiene CSS ni HTML manual.

Estructura
──────────
 1. Configuración de página
 2. Carga de datos y modelos        → cargar_datos() / cargar_modelos()   [sin cambios]
 3. Lógica de negocio               → calcular_plan_reposicion()          [sin cambios]
 4. Helpers de gráficos Plotly      → layout_grafico() / eje()
 5. VISTA USUARIO (personal)        → vista_usuario()
 6. VISTA DESARROLLADOR / ANALISTA  → vista_desarrollador()
 7. Enrutado principal              → main()
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.inspection import permutation_importance

# Compatibilidad con modelos serializados en scikit-learn 1.6.x.
# El artefacto binario fue generado con una versión que aún exponía
# _RemainderColsList dentro de sklearn.compose._column_transformer.
try:
    from sklearn.compose import _column_transformer as _ct_mod

    if not hasattr(_ct_mod, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        _ct_mod._RemainderColsList = _RemainderColsList
except Exception:
    pass

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN DE PÁGINA
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ALDIMI Predict",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE = os.getcwd()


# ══════════════════════════════════════════════════════════════════════════════
# 2. CARGA DE DATOS Y MODELOS  (lógica intacta)
#    Contratos:
#      cargar_datos()   → (df_inv, df_orig, df_pac, df_catalogo, fuente:str)
#      cargar_modelos() → (m_inv:dict, m_pac:dict)
# ══════════════════════════════════════════════════════════════════════════════
DB_COMUN = os.path.join(BASE, "data", "aldimi_core.db")


@st.cache_data
def cargar_datos():
    """Lee de la BD común SQLite; si no existe o está desactualizada, usa los CSV."""
    if os.path.exists(DB_COMUN):
        import sqlite3
        con = sqlite3.connect(DB_COMUN)
        try:
            tablas = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            if {"inventario_semanal", "articulos", "pacientes", "catalogo_productos"} <= tablas:
                df_inv  = pd.read_sql("SELECT * FROM inventario_semanal", con)
                df_orig = pd.read_sql("SELECT * FROM articulos", con)
                df_pac  = pd.read_sql("SELECT * FROM pacientes", con)
                df_cat  = pd.read_sql("SELECT * FROM catalogo_productos", con)
                if "nombre_producto" in df_inv.columns:
                    return df_inv, df_orig, df_pac, df_cat, "BD común (aldimi_core.db)"
        finally:
            con.close()
    df_inv  = pd.read_csv(os.path.join(BASE, "data", "processed", "aldimi_dataset_semanal.csv"))
    df_orig = pd.read_csv(os.path.join(BASE, "data", "processed", "aldimi_dataset_completo.csv"))
    df_pac  = pd.read_csv(os.path.join(BASE, "data", "processed", "aldimi_pacientes_sintetico.csv"))
    df_cat  = pd.read_csv(os.path.join(BASE, "data", "processed", "catalogo_productos.csv"))
    return df_inv, df_orig, df_pac, df_cat, "CSV locales (ejecuta integracion_bd.py para actualizar la BD)"


@st.cache_resource
def cargar_modelos():
    """Carga los pipelines serializados de inventario y pacientes."""
    m_inv = joblib.load(os.path.join(BASE, "models", "models_inventario.pkl"))
    m_pac = joblib.load(os.path.join(BASE, "models", "risk_model_rf_binary.joblib"))
    return m_inv, m_pac


PACIENTE_ID_COL = "id_paciente"
PACIENTE_TARGET_COL = "nivel_riesgo"
BAJO_MAX = 0.35
ALTO_MIN = 0.67


def banda_desde_score(score: float) -> str:
    if score < BAJO_MAX:
        return "Bajo"
    if score <= ALTO_MIN:
        return "Revisión"
    return "Alto"


def columnas_paciente_modelo(df_pac: pd.DataFrame, modelo) -> list[str]:
    """Devuelve las columnas de entrada reales que espera el pipeline de pacientes."""
    candidatas = [
        c for c in getattr(modelo, "feature_names_in_", [])
        if c in df_pac.columns and c not in {PACIENTE_ID_COL, PACIENTE_TARGET_COL}
    ]
    if candidatas:
        return list(candidatas)

    if hasattr(modelo, "named_steps"):
        for step in reversed(list(modelo.named_steps.values())):
            if hasattr(step, "feature_names_in_"):
                candidatas = [
                    c for c in list(step.feature_names_in_)
                    if c in df_pac.columns and c not in {PACIENTE_ID_COL, PACIENTE_TARGET_COL}
                ]
                if candidatas:
                    return candidatas

    return [c for c in df_pac.columns if c not in {PACIENTE_ID_COL, PACIENTE_TARGET_COL}]


def clasificador_paciente(modelo):
    """Obtiene el estimador final de un pipeline o devuelve el objeto recibido."""
    if hasattr(modelo, "named_steps"):
        return list(modelo.named_steps.values())[-1]
    return modelo


def score_paciente_alto(modelo, X: pd.DataFrame) -> np.ndarray:
    """Score de riesgo alto con tolerancia a distintos formatos de classes_."""
    probs = modelo.predict_proba(X)
    clases = list(getattr(modelo, "classes_", []))
    if not clases and hasattr(modelo, "named_steps"):
        ultimo = clasificador_paciente(modelo)
        clases = list(getattr(ultimo, "classes_", []))

    if 1 in clases:
        idx_alto = clases.index(1)
    elif "Alto" in clases:
        idx_alto = clases.index("Alto")
    elif len(clases) > 1:
        idx_alto = 1
    else:
        idx_alto = 0
    return probs[:, idx_alto]


def pacientes_con_score(df_pac: pd.DataFrame, modelo) -> tuple[pd.DataFrame, list[str]]:
    """Anota el dataframe de pacientes con score y banda operacional del modelo."""
    columnas = columnas_paciente_modelo(df_pac, modelo)
    X = df_pac[columnas].copy()
    out = df_pac.copy()
    out["risk_score"] = score_paciente_alto(modelo, X)
    out["banda_riesgo"] = out["risk_score"].apply(banda_desde_score)
    return out, columnas


# ══════════════════════════════════════════════════════════════════════════════
# 3. LÓGICA DE NEGOCIO  (intacta)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def calcular_plan_reposicion(df_inv: pd.DataFrame, df_catalogo: pd.DataFrame,
                             _m_inv: dict) -> pd.DataFrame:
    """Predice riesgo de quiebre a 7/14 días sobre la última semana de cada producto real."""
    catalogo = df_catalogo.set_index("codigo_articulo")
    ult = (df_inv.sort_values("semana_del_año")
                 .groupby("codigo_articulo", as_index=False).tail(1).copy())
    ult = ult[ult["codigo_articulo"].isin(
        catalogo.index[catalogo["es_producto"].astype(bool)])].copy()

    X = np.column_stack([
        _m_inv["le_categoria"].transform(ult["categoria"]),
        ult["ocupacion_albergue"],
        ult["stock_fin_semana"],
        np.zeros(len(ult)),                       # sin ingresos confirmados
        ult["rolling_avg_salidas_3sem"],
        ult["rolling_avg_salidas_3sem"],
        np.minimum(ult["semana_del_año"] + 1, 52),
    ])
    X_sc   = _m_inv["scaler"].transform(X)
    prob7  = _m_inv["modelo_7d"].predict_proba(X_sc)[:, 1]
    prob14 = _m_inv["modelo_14d"].predict_proba(X_sc)[:, 1]

    stock = ult["stock_fin_semana"].values
    info  = catalogo.loc[ult["codigo_articulo"]]
    plan = pd.DataFrame({
        "Producto":        info["nombre_producto"].values,
        "Categoría":       info["categoria_general"].values,
        "Unidad":          info["unidad_medida"].values,
        "Stock actual":    np.maximum(ult["stock_fin_semana"].round(1).values, 0),
        "Consumo semanal": ult["rolling_avg_salidas_3sem"].round(1).values,
        "Riesgo 7 días":   (prob7 * 100).round(0).astype(int),
        "Riesgo 14 días":  (prob14 * 100).round(0).astype(int),
        "Código":          ult["codigo_articulo"].values,
    })
    plan["Estado"] = np.select(
        [stock <= 0, prob7 >= 0.5, prob14 >= 0.5],
        ["Agotado", "Crítico (7 días)", "Planificar (14 días)"],
        default="Cubierto")
    orden = plan["Estado"].map({"Agotado": 0, "Crítico (7 días)": 1,
                                "Planificar (14 días)": 2, "Cubierto": 3})
    return (plan.assign(_o=orden)
                .sort_values(["_o", "Consumo semanal"], ascending=[True, False])
                .drop(columns="_o"))


# ══════════════════════════════════════════════════════════════════════════════
# 4. HELPERS DE GRÁFICOS PLOTLY
#    Fondos transparentes: los gráficos heredan el tema de .streamlit/config.toml
# ══════════════════════════════════════════════════════════════════════════════
def layout_grafico(alto: int = 320, leyenda: bool = False) -> dict:
    return dict(height=alto,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(size=12, color="#F8FAFC"),
                margin=dict(l=6, r=6, t=28, b=6),
                showlegend=leyenda)


def eje(titulo: str = "", grid: bool = True) -> dict:
    return dict(title=dict(text=titulo, font=dict(size=11)),
                tickfont=dict(size=11, color="#CBD5E1"),
                showgrid=grid, gridcolor="rgba(148,163,184,0.18)",
                zeroline=False, showline=False, linecolor="rgba(148,163,184,0.25)")


# ══════════════════════════════════════════════════════════════════════════════
# 5. VISTA USUARIO — Personal del albergue
#    Minimalista y operativa. Solo componentes nativos: metric, error/warning,
#    dataframe, progress, expander, container.
# ══════════════════════════════════════════════════════════════════════════════
def vista_usuario(ctx: dict) -> None:
    plan, df_pac = ctx["plan"], ctx["df_pac"]
    m_pac = ctx["m_pac"]
    df_pac, _ = pacientes_con_score(df_pac, m_pac)

    agotados   = plan[plan["Estado"] == "Agotado"]
    criticos   = plan[plan["Estado"] == "Crítico (7 días)"]
    planificar = plan[plan["Estado"] == "Planificar (14 días)"]
    pac_alto = df_pac[df_pac["banda_riesgo"] == "Alto"]

    st.title("Panel de gestión del albergue")
    st.caption(f"Semana {ctx['semana_actual']} · {ctx['ocupacion_actual']} familias alojadas · "
               "Lo importante primero: qué reponer y a quién atender.")

    tab_resumen, tab_inv, tab_pac = st.tabs(["Resumen Operativo", "Inventario", "Pacientes"])

    # ── RESUMEN OPERATIVO ─────────────────────────────────────────────────────
    with tab_resumen:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Productos agotados", len(agotados),
                  delta="Reponer de inmediato" if len(agotados) else "Sin pendientes",
                  delta_color="inverse" if len(agotados) else "normal")
        c2.metric("Críticos a 7 días", len(criticos),
                  delta="Atención urgente" if len(criticos) else "Sin pendientes",
                  delta_color="inverse" if len(criticos) else "normal")
        c3.metric("Pacientes prioridad alta", len(pac_alto),
                  delta="Atender primero" if len(pac_alto) else "Sin casos",
                  delta_color="inverse" if len(pac_alto) else "normal")
        c4.metric("Familias alojadas", ctx["ocupacion_actual"],
                  delta="Capacidad: 100", delta_color="off")

        st.divider()
        col_a, col_b = st.columns(2, gap="medium")

        with col_a:
            with st.container(border=True):
                st.subheader("Reposición urgente — esta semana")
                urgentes = pd.concat([agotados, criticos])
                if urgentes.empty:
                    st.success("Sin productos en riesgo esta semana.")
                else:
                    for _, r in urgentes.head(6).iterrows():
                        detalle = ("sin stock" if r["Estado"] == "Agotado"
                                   else f"quedan {r['Stock actual']:g} {r['Unidad']}")
                        st.error(f"**{r['Producto']}** · {r['Categoría']} — {detalle} · "
                                 f"consumo ~{r['Consumo semanal']:g} {r['Unidad']}/sem · "
                                 f"riesgo 7 días: **{r['Riesgo 7 días']}%**")
                    if len(urgentes) > 6:
                        st.caption(f"… y {len(urgentes) - 6} más en la pestaña Inventario.")

        with col_b:
            with st.container(border=True):
                st.subheader("Pacientes con prioridad alta")
                if pac_alto.empty:
                    st.success("Sin pacientes clasificados con prioridad alta.")
                else:
                    for _, p in pac_alto.head(6).iterrows():
                        senales = []
                        if p["etapa_cancer"] in ("III", "IV"):  senales.append(f"etapa {p['etapa_cancer']}")
                        if p["presencia_infeccion"] == 1:       senales.append("infección")
                        if p["estado_nutricional"] != "Normal": senales.append("desnutrición")
                        if p["perdida_peso_reciente"] == 1:     senales.append("pérdida de peso")
                        st.error(f"**{p['id_paciente']}** · {int(p['edad'])} años · "
                                 f"{p['diagnostico']} — score {p['risk_score']:.2f} · señales: "
                                 f"{', '.join(senales[:3]) or 'revisar historia'}")
                    if len(pac_alto) > 6:
                        st.caption(f"… y {len(pac_alto) - 6} más en la pestaña Pacientes.")

    # ── INVENTARIO ────────────────────────────────────────────────────────────
    with tab_inv:
        st.header("Plan de reposición — próximos 14 días")
        f1, f2, f3 = st.columns([2, 2, 1], gap="medium")
        with f1:
            filtro = st.radio("Mostrar", ["Solo con acción pendiente", "Todos los productos"],
                              horizontal=True, label_visibility="collapsed")
        with f2:
            busqueda = st.text_input("Buscar", placeholder="Buscar por nombre o código…",
                                     label_visibility="collapsed")
        vista = plan if filtro == "Todos los productos" else plan[plan["Estado"] != "Cubierto"]
        if busqueda.strip():
            q = busqueda.strip().lower()
            vista = vista[vista["Producto"].str.lower().str.contains(q, regex=False) |
                          vista["Código"].str.lower().str.contains(q, regex=False)]
        with f3:
            st.download_button("Descargar lista (CSV)",
                               vista.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"plan_reposicion_semana{ctx['semana_actual']}.csv",
                               mime="text/csv", use_container_width=True)

        if vista.empty:
            st.success("Todo el almacén está cubierto para las próximas dos semanas."
                       if not busqueda.strip() else "Sin resultados para esa búsqueda.")
        else:
            st.dataframe(
                vista, use_container_width=True, hide_index=True, height=340,
                column_config={
                    "Producto": st.column_config.TextColumn("Producto", width="large"),
                    "Riesgo 7 días": st.column_config.ProgressColumn(
                        "Riesgo 7 días", format="%d%%", min_value=0, max_value=100),
                    "Riesgo 14 días": st.column_config.ProgressColumn(
                        "Riesgo 14 días", format="%d%%", min_value=0, max_value=100),
                })
            st.caption(f"{len(agotados)} agotados · {len(criticos)} críticos · "
                       f"{len(planificar)} para planificar. El riesgo es la probabilidad "
                       "de quedarse sin stock estimada por el sistema.")

        with st.expander("Consultar un producto — ¿alcanzará el stock?"):
            m_inv = ctx["m_inv"]
            s1, s2, s3 = st.columns([2, 2, 1], gap="medium")
            with s1:
                cat_sel = st.selectbox("Tipo de producto", m_inv["categorias"])
                unidad  = ctx["unidad_por_tipo"].get(cat_sel, "unidades")
                stock_h = st.number_input(f"Stock disponible hoy ({unidad})", 0.0, 500.0, 8.0, 1.0)
            with s2:
                consumo = st.number_input(f"Consumo semanal aproximado ({unidad})", 0.0, 100.0, 3.0, 0.5)
                ocupa   = st.slider("Familias en el albergue", 40, 100, ctx["ocupacion_actual"])
            with s3:
                evaluar = st.button("Evaluar", use_container_width=True)
            if evaluar:
                X = np.array([[m_inv["le_categoria"].transform([cat_sel])[0],
                               ocupa, stock_h, 0.0, consumo, consumo, ctx["semana_actual"]]])
                X_sc = m_inv["scaler"].transform(X)
                for dias, mod in [(7, "modelo_7d"), (14, "modelo_14d")]:
                    prob = m_inv[mod].predict_proba(X_sc)[0][1]
                    pred = m_inv[mod].predict(X_sc)[0]
                    if pred:
                        st.error(f"**A {dias} días — se agotaría** ({prob*100:.0f}% de riesgo). "
                                 "Pedir reposición o donación esta misma semana.")
                    else:
                        st.success(f"**A {dias} días — alcanza** ({prob*100:.0f}% de riesgo). "
                                   "El stock cubre este horizonte.")

    # ── PACIENTES ─────────────────────────────────────────────────────────────
    with tab_pac:
        st.header("Pacientes que requieren atención primero")
        if pac_alto.empty:
            st.success("Ningún paciente con score de riesgo en banda Alta.")
        else:
            tabla = pac_alto[["id_paciente", "edad", "diagnostico", "etapa_cancer",
                              "estado_nutricional", "presencia_infeccion",
                              "adherencia_tratamiento", "risk_score"]].copy()
            tabla["presencia_infeccion"] = tabla["presencia_infeccion"].map({1: "Sí", 0: "No"})
            tabla.columns = ["Paciente", "Edad", "Diagnóstico", "Etapa",
                             "Nutrición", "Infección", "Sigue tratamiento", "Score"]
            st.dataframe(
                tabla, use_container_width=True, hide_index=True, height=280,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score", format="%.2f", min_value=0.0, max_value=1.0)
                },
            )
            st.caption(f"{len(pac_alto)} de {len(df_pac)} pacientes (score > {ALTO_MIN:.2f}). "
                       "Confirmar cada caso con el equipo médico.")

        st.divider()
        st.subheader("Situación general (score del modelo)")
        for banda, etiqueta in [("Alto", "Prioridad alta"), ("Revisión", "En revisión"),
                                ("Bajo", "Prioridad baja")]:
            cnt = int((df_pac["banda_riesgo"] == banda).sum())
            pct = cnt / len(df_pac)
            st.progress(pct, text=f"{etiqueta}: {cnt} pacientes ({pct*100:.0f}%)")

        with st.expander("Ver etiqueta original del dataset (referencia, no usada por el modelo)"):
            st.caption(
                "La columna `nivel_riesgo` proviene del dataset sintético original (3 clases) "
                "y se muestra únicamente como referencia / comparación. El sistema actual "
                "clasifica y prioriza usando el score binario del modelo, no esta etiqueta."
            )
            tabla_ref = df_pac[["id_paciente", "nivel_riesgo", "banda_riesgo", "risk_score"]].copy()
            tabla_ref.columns = ["Paciente", "Etiqueta original (CSV)", "Banda del modelo", "Score"]
            st.dataframe(tabla_ref, use_container_width=True, hide_index=True, height=240)

        st.divider()
        with st.expander("Evaluar prioridad de un paciente nuevo"):
            formulario_paciente(ctx)


def formulario_paciente(ctx: dict) -> None:
    """Formulario de evaluación de prioridad (lógica de encoding intacta)."""
    m_pac = ctx["m_pac"]
    with st.form("form_paciente"):
        g1, g2, g3 = st.columns(3)
        with g1:
            edad      = st.number_input("Edad (años)", 2, 17, 8)
            sexo      = st.selectbox("Sexo", ["Masculino", "Femenino"])
            distancia = st.number_input("Distancia a Lima (km)", 10, 1400, 600)
            lugar     = st.selectbox("Procedencia", ["Sierra sur", "Sierra norte", "Sierra centro",
                                                     "Selva", "Costa norte", "Costa sur", "Lima"])
            instruccion = st.selectbox("Instrucción del cuidador",
                                       ["Sin estudios", "Primaria", "Secundaria",
                                        "Superior técnica", "Superior universitaria"])
        with g2:
            diagnostico = st.selectbox("Diagnóstico", [
                "Leucemia linfoblástica aguda", "Leucemia mieloide aguda",
                "Linfoma de Hodgkin", "Linfoma no Hodgkin", "Tumor cerebral",
                "Neuroblastoma", "Tumor de Wilms", "Osteosarcoma",
                "Retinoblastoma", "Rabdomiosarcoma"])
            etapa       = st.selectbox("Etapa del cáncer", ["I", "II", "III", "IV"])
            tratamiento = st.selectbox("Tratamiento", ["Quimioterapia", "Radioterapia", "Cirugía",
                                                       "Quimio + Radio", "Quimio + Cirugía"])
            meses_trat  = st.number_input("Meses en tratamiento", 1, 36, 6)
            motivo_ing  = st.selectbox("Motivo de ingreso", ["Tratamiento", "Control", "Examen", "Emergencia"])
            motivo_re   = st.selectbox("Motivo de reingreso", [
                "Primer ingreso", "Continuación de tratamiento", "Recaída",
                "Continuación de quimioterapia", "Seguimiento médico",
                "Continuación de radioterapia", "Control médico"])
        with g3:
            estado_fis = st.selectbox("Movilidad", ["Caminando", "Con ayuda parcial",
                                                    "Permanece en cama", "Usa silla de ruedas"])
            estado_nut = st.selectbox("Estado nutricional", ["Normal", "Desnutrición leve", "Desnutrición severa"])
            infeccion  = st.selectbox("¿Infección activa?", ["No", "Sí"])
            peso       = st.selectbox("¿Pérdida de peso reciente?", ["No", "Sí"])
            adherencia = st.selectbox("¿Sigue el tratamiento?", ["Alta", "Media", "Baja"])
            apoyo      = st.selectbox("Apoyo familiar", ["Fuerte", "Moderado", "Limitado"])
            acceso     = st.selectbox("Acceso a medicamentos", ["Completo", "Parcial", "Limitado"])
            emocional  = st.selectbox("Estado emocional", ["Estable", "Ansioso", "Deprimido"])
            comorb     = st.number_input("Otras enfermedades (n°)", 0, 3, 0)
            frec_hosp  = st.number_input("Hospitalizaciones (últimos 3 meses)", 0, 8, 0)
            reingresos = st.number_input("N° reingresos previos", 0, 7, 0)
        enviado = st.form_submit_button("Evaluar prioridad de atención",
                                        use_container_width=True)

    if not enviado:
        return

    X_new = pd.DataFrame([{
        "edad": edad,
        "sexo": sexo,
        "lugar_procedencia": lugar,
        "distancia_origen_km": distancia,
        "grado_instruccion_cuidador": instruccion,
        "diagnostico": diagnostico,
        "etapa_cancer": etapa,
        "tipo_tratamiento": tratamiento,
        "meses_en_tratamiento": meses_trat,
        "num_reingresos": reingresos,
        "motivo_ingreso": motivo_ing,
        "motivo_reingreso": motivo_re,
        "estado_fisico": estado_fis,
        "estado_nutricional": estado_nut,
        "presencia_infeccion": int(infeccion == "Sí"),
        "frecuencia_hospitalizacion_3m": frec_hosp,
        "adherencia_tratamiento": adherencia,
        "apoyo_familiar": apoyo,
        "acceso_medicamentos": acceso,
        "estado_emocional_paciente": emocional,
        "perdida_peso_reciente": int(peso == "Sí"),
        "num_comorbilidades": comorb,
    }])

    # --- Binary risk score (Alto vs Bajo), Medio ya no es una clase entrenada ---
    score = score_paciente_alto(m_pac, X_new)[0]

    if score < BAJO_MAX:
        banda = "Bajo"
    elif score <= ALTO_MIN:
        banda = "Revisión"
    else:
        banda = "Alto"

    if banda == "Alto":
        st.error(f"**PRIORIDAD ALTA** (score: {score:.2f}) — Atender primero: avisar al "
                 "médico tratante y hacer seguimiento diario.")
    elif banda == "Revisión":
        st.warning(f"**ZONA DE REVISIÓN** (score: {score:.2f}) — Caso ambiguo: se recomienda "
                   "evaluación clínica adicional antes de definir prioridad.")
    else:
        st.success(f"**PRIORIDAD BAJA** (score: {score:.2f}) — Continuar con los controles "
                   "habituales según cronograma.")

    st.caption(
        f"Score de riesgo: {score:.2f} sobre 1.00 (0 = bajo riesgo, 1 = alto riesgo). "
        f"Bandas: Bajo < {BAJO_MAX:.2f} · Revisión {BAJO_MAX:.2f}–{ALTO_MIN:.2f} · "
        f"Alto > {ALTO_MIN:.2f}. Resultado orientativo; la evaluación clínica la "
        "confirma el personal de salud."
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6. VISTA DESARROLLADOR / ANALISTA
# ══════════════════════════════════════════════════════════════════════════════
def vista_desarrollador(ctx: dict) -> None:
    df_inv, df_orig, df_pac = ctx["df_inv"], ctx["df_orig"], ctx["df_pac"]
    m_inv, m_pac = ctx["m_inv"], ctx["m_pac"]
    df_pac_modelo = ctx["df_pac_modelo"]
    paciente_cols = ctx["paciente_feature_cols"]

    st.title("Consola técnica — modelos y datos")
    st.caption(f"Fuente: {ctx['fuente_datos']} · {len(df_inv):,} registros semanales · "
               f"{len(df_pac)} pacientes")

    tab_mod, tab_pipe, tab_feat, tab_dist = st.tabs(
        ["Rendimiento de modelos", "Pipeline de datos", "Análisis de features", "Distribuciones"])

    # ── MODELOS ───────────────────────────────────────────────────────────────
    with tab_mod:
        col_i, col_p = st.columns(2, gap="large")

        with col_i:
            with st.container(border=True):
                st.subheader("Inventario — clasificación binaria")
                met_inv = m_inv.get("metricas_7d", {})
                if met_inv:
                    df_m = pd.DataFrame(met_inv).T
                    df_m.columns = ["F1-Score", "AUC-ROC"][:len(df_m.columns)]
                    st.dataframe(df_m.round(4), use_container_width=True)
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=df_m.index, y=df_m["F1-Score"], name="F1"))
                    if "AUC-ROC" in df_m.columns:
                        fig.add_trace(go.Bar(x=df_m.index, y=df_m["AUC-ROC"], name="AUC"))
                    fig.update_layout(**layout_grafico(260, leyenda=True), barmode="group",
                                      yaxis=eje("Score") | {"range": [0, 1.05]}, xaxis=eje())
                    st.plotly_chart(fig, use_container_width=True)
                st.caption(f"En producción — 7 días: {m_inv.get('mejor_nombre_7d', '–')} · "
                           f"14 días: {m_inv.get('mejor_nombre_14d', '–')}. "
                           "CV estratificada (5 folds) + RandomizedSearchCV.")

        with col_p:
            with st.container(border=True):
                st.subheader("Pacientes — clasificación binaria")
                modelo_paciente = clasificador_paciente(m_pac)
                if hasattr(modelo_paciente, "feature_importances_"):
                    df_bin = df_pac_modelo[df_pac_modelo["nivel_riesgo"].isin(["Bajo", "Alto"])]
                    y_bin = (df_bin["nivel_riesgo"] == "Alto").astype(int)
                    if len(df_bin) >= 20 and y_bin.nunique() > 1:
                        imp_p = permutation_importance(
                            m_pac,
                            df_bin[paciente_cols],
                            y_bin,
                            n_repeats=10,
                            random_state=42,
                            scoring="f1",
                        )
                        imp_ser = (pd.Series(imp_p.importances_mean, index=paciente_cols)
                                     .sort_values())
                        fig = go.Figure(go.Bar(x=imp_ser.values, y=imp_ser.index, orientation="h"))
                        fig.update_layout(**layout_grafico(300), xaxis=eje("Importancia"),
                                          yaxis=eje(grid=False) | {"automargin": True})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No hay suficientes filas Alto/Bajo para estimar importancia por permutación.")
                else:
                    st.info("El modelo binario no expone feature_importances_ y no se calculó importancia por permutación.")
                st.caption(f"En producción: modelo binario RF · {len(paciente_cols)} variables de entrada · "
                           "score de riesgo alto con bandas Bajo / Revisión / Alto.")

    # ── PIPELINE ──────────────────────────────────────────────────────────────
    with tab_pipe:
        st.header("Origen y confluencia de datos (SQLite)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("inventario_semanal", f"{len(df_inv):,}", delta="registros semanales",
                  delta_color="off")
        c2.metric("articulos", f"{len(df_orig):,}", delta="artículos históricos",
                  delta_color="off")
        c3.metric("pacientes", len(df_pac), delta="registros clínicos", delta_color="off")
        c4.metric("catalogo_productos", len(ctx["df_catalogo"]), delta="productos catalogados",
                  delta_color="off")
        st.info(f"Fuente activa: **{ctx['fuente_datos']}**. La BD común `aldimi_core.db` "
                "se regenera con `integracion_bd.py`; si no existe, la app recurre a los "
                "CSV de `data/processed/`.")

        with st.expander("Muestras de datos crudos"):
            sel = st.selectbox("Tabla", ["inventario_semanal", "articulos",
                                         "pacientes", "catalogo_productos"])
            df_sel = {"inventario_semanal": df_inv, "articulos": df_orig,
                      "pacientes": df_pac, "catalogo_productos": ctx["df_catalogo"]}[sel]
            st.dataframe(df_sel.head(50), use_container_width=True, height=320)
            st.caption(f"{df_sel.shape[0]:,} filas × {df_sel.shape[1]} columnas · "
                       f"tipos: {df_sel.dtypes.value_counts().to_dict()}")

    # ── FEATURES ──────────────────────────────────────────────────────────────
    with tab_feat:
        FEATS_INV = ["categoria", "ocupacion_albergue", "stock_inicio", "ingresos",
                     "salidas", "rolling_avg_3sem", "semana_del_año"]

        with st.container(border=True):
            st.subheader("Importancia de features — inventario (7 días)")
            modelo7 = m_inv["modelo_7d"]
            if hasattr(modelo7, "feature_importances_"):
                imp = pd.Series(modelo7.feature_importances_,
                                index=FEATS_INV[:len(modelo7.feature_importances_)]).sort_values()
                fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h"))
                fig.update_layout(**layout_grafico(460), xaxis=eje("Importancia"),
                                  yaxis=eje(grid=False) | {"automargin": True})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("El modelo en producción no expone feature_importances_.")

        with st.container(border=True):
            st.subheader("Importancia de features — pacientes (top 15)")
            modelo_p = m_pac["modelo"]
            if hasattr(modelo_p, "feature_importances_"):
                imp_p = (pd.Series(modelo_p.feature_importances_, index=m_pac["feat_cols"])
                         .nlargest(15).sort_values())
                fig = go.Figure(go.Bar(x=imp_p.values, y=imp_p.index, orientation="h"))
                fig.update_layout(**layout_grafico(520), xaxis=eje("Importancia"),
                                  yaxis=eje(grid=False) | {"automargin": True,
                                                           "tickfont": dict(size=11)})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("El modelo en producción no expone feature_importances_.")

    # ── DISTRIBUCIONES ────────────────────────────────────────────────────────
    with tab_dist:
        col_ev, col_top = st.columns([3, 2], gap="medium")

        with col_ev:
            st.subheader("Evolución anual de stock por categoría")
            cat_sel = st.selectbox("Categoría general",
                                   sorted(df_inv["categoria_general"].dropna().unique()))
            df_cat = (df_inv[df_inv["categoria_general"] == cat_sel]
                      .groupby("semana_del_año")
                      .agg(stock_promedio=("stock_fin_semana", "mean"),
                           alertas=("alerta_7_dias", "sum")).reset_index())
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=df_cat["semana_del_año"], y=df_cat["stock_promedio"],
                                     name="Stock promedio", mode="lines", fill="tozeroy"),
                          secondary_y=False)
            fig.add_trace(go.Bar(x=df_cat["semana_del_año"], y=df_cat["alertas"],
                                 name="Semanas con alerta", opacity=0.45), secondary_y=True)
            fig.update_layout(**layout_grafico(320, leyenda=True),
                              xaxis=eje("Semana del año"),
                              legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10)))
            fig.update_yaxes(title_text="Stock", secondary_y=False,
                             gridcolor="rgba(128,128,128,0.25)")
            fig.update_yaxes(title_text="Alertas", secondary_y=True, showgrid=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_top:
            st.subheader("Tasa de rotación — top 12")
            top = (df_orig[df_orig["es_producto"].astype(bool)]
                   .nlargest(12, "tasa_rotacion")[["nombre_producto", "tasa_rotacion"]])
            fig = go.Figure(go.Bar(x=top["tasa_rotacion"],
                                   y=top["nombre_producto"].str.slice(0, 28),
                                   orientation="h"))
            fig.update_layout(**layout_grafico(320), xaxis=eje("Rotación"),
                              yaxis=eje(grid=False) | {"automargin": True,
                                                       "tickfont": dict(size=9)})
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Distribución de pacientes")
        d1, d2 = st.columns(2, gap="medium")
        with d1:
            vc = df_pac["nivel_riesgo"].value_counts().reindex(["Bajo", "Medio", "Alto"])
            fig = go.Figure(go.Pie(labels=vc.index, values=vc.values, hole=0.55,
                                   textinfo="label+percent", textfont=dict(size=11)))
            fig.update_layout(**layout_grafico(280))
            st.plotly_chart(fig, use_container_width=True)
        with d2:
            fig = go.Figure(go.Histogram(x=df_pac["edad"], nbinsx=16))
            fig.update_layout(**layout_grafico(280), xaxis=eje("Edad (años)"),
                              yaxis=eje("Pacientes"))
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 7. ENRUTADO PRINCIPAL  (intacto: divide vista usuario / desarrollador)
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    # ── Carga (datos y modelos entran aquí, una sola vez) ────────────────────
    try:
        df_inv, df_orig, df_pac, df_catalogo, fuente = cargar_datos()
        m_inv, m_pac = cargar_modelos()
    except FileNotFoundError as e:
        st.error(f"Archivos no encontrados: {e}")
        st.stop()

    plan = calcular_plan_reposicion(df_inv, df_catalogo, m_inv)
    df_pac_modelo, paciente_cols = pacientes_con_score(df_pac, m_pac)
    ctx = {
        "df_inv": df_inv, "df_orig": df_orig, "df_pac": df_pac,
        "df_pac_modelo": df_pac_modelo,
        "paciente_feature_cols": paciente_cols,
        "df_catalogo": df_catalogo, "fuente_datos": fuente,
        "m_inv": m_inv, "m_pac": m_pac, "plan": plan,
        "ocupacion_actual": int(df_inv.loc[df_inv["semana_del_año"].idxmax(),
                                           "ocupacion_albergue"]),
        "semana_actual": int(df_inv["semana_del_año"].max()),
        "unidad_por_tipo": (df_catalogo.groupby("categoria")["unidad_medida"]
                            .agg(lambda s: s.mode().iat[0]).to_dict()),
    }

    # ── Sidebar: identidad + selector de rol ─────────────────────────────────
    with st.sidebar:
        st.title("ALDIMI Predict")
        st.caption("Albergue Divina Misericordia")
        st.divider()

        modo = st.radio("Modo de visualización",
                        ["Personal del albergue", "Desarrollador / Analista"])
        st.divider()

        st.subheader("Resumen de hoy")
        agotados = int((plan["Estado"] == "Agotado").sum())
        criticos = int((plan["Estado"] == "Crítico (7 días)").sum())
        alta     = int((df_pac_modelo["banda_riesgo"] == "Alto").sum())
        st.markdown(
            f"- **{agotados}** productos agotados\n"
            f"- **{criticos}** críticos a 7 días\n"
            f"- **{alta}** pacientes prioridad alta\n"
            f"- **{ctx['ocupacion_actual']}** familias alojadas"
        )
        st.divider()
        st.caption("Las recomendaciones apoyan la decisión del equipo; no reemplazan "
                   "el criterio del personal de salud.")
        st.caption("ML 1ACC0057 · UPC · Julio 2026")

    if modo == "Personal del albergue":
        vista_usuario(ctx)
    else:
        vista_desarrollador(ctx)


main()
