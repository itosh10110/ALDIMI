"""
ALDIMI Predict — Dashboard 100% componentes nativos de Streamlit.

El tema visual (colores, fondo, tipografía) se define en `.streamlit/config.toml`;
este archivo no contiene CSS ni HTML manual.

Actualización (6 jul): feedback de Jairo — la vista de personal (vista_usuario)
ya no muestra gráficos, scores ni porcentajes de confianza; el consumo de
inventario se recalcula en vivo según el número de familias que el personal
ingresa (ver calcular_cobertura_por_familias()); se agregaron formularios para
registrar ingresos de inventario y para dar de alta/editar pacientes. Las
métricas técnicas (scores, importancia de variables, gráficos) siguen
disponibles íntegras en vista_desarrollador() ("Modo Desarrollador").

Estructura
──────────
 1. Configuración de página
 2. Carga de datos y modelos        → cargar_datos() / cargar_modelos()   [sin cambios]
 3. Lógica de negocio               → calcular_plan_reposicion()          [sin cambios]
    + cobertura dinámica por familias, registro de ingresos y de pacientes (nuevo)
 4. Helpers de gráficos Plotly      → layout_grafico() / eje()
 5. VISTA USUARIO (personal)        → vista_usuario()                    [sin tecnicismos]
 6. VISTA DESARROLLADOR / ANALISTA  → vista_desarrollador()               [sin cambios]
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
MOVIMIENTOS_CSV = os.path.join(BASE, "data", "processed", "movimientos_inventario.csv")
PACIENTES_CSV = os.path.join(BASE, "data", "processed", "aldimi_pacientes_sintetico.csv")


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
# 3bis. COBERTURA DINÁMICA POR NÚMERO DE FAMILIAS  (nuevo, feedback de Jairo 6 jul)
#    En vez de extrapolar el clasificador fuera de su rango de entrenamiento
#    (40–60 familias; ver notebooks/modelo_cantidad_inventario.ipynb, sección 7),
#    se usa una tasa de consumo histórica POR FAMILIA para cada producto. Así el
#    "cuándo se acaba" escala de forma transparente y monotónica con las familias
#    que ingresa el personal, sin depender de una caja negra.
# ══════════════════════════════════════════════════════════════════════════════
def calcular_tasa_percapita(df_inv: pd.DataFrame, df_catalogo: pd.DataFrame) -> pd.Series:
    """Consumo semanal promedio por familia (histórico), por producto real.
    Usa rolling_avg_salidas_3sem para ser consistente con "Consumo semanal"
    del plan de reposición."""
    catalogo = df_catalogo.set_index("codigo_articulo")
    reales = catalogo.index[catalogo["es_producto"].astype(bool)]
    real = df_inv[df_inv["codigo_articulo"].isin(reales)].copy()
    real["percapita"] = real["rolling_avg_salidas_3sem"] / real["ocupacion_albergue"].replace(0, np.nan)
    return real.groupby("codigo_articulo")["percapita"].mean().fillna(0.0)


def cargar_movimientos_inventario() -> pd.DataFrame:
    """Ingresos de producto registrados a mano por el personal (no viene en el histórico)."""
    cols = ["fecha", "codigo_articulo", "cantidad", "nota"]
    if os.path.exists(MOVIMIENTOS_CSV):
        return pd.read_csv(MOVIMIENTOS_CSV)
    return pd.DataFrame(columns=cols)


def guardar_ingreso_inventario(codigo: str, cantidad: float, nota: str = "") -> None:
    """Registra la llegada de un producto y refresca la caché para que se
    refleje de inmediato en el stock y en la cobertura."""
    from datetime import datetime
    df_mov = cargar_movimientos_inventario()
    nueva = pd.DataFrame([{
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "codigo_articulo": codigo, "cantidad": cantidad, "nota": nota,
    }])
    os.makedirs(os.path.dirname(MOVIMIENTOS_CSV), exist_ok=True)
    pd.concat([df_mov, nueva], ignore_index=True).to_csv(MOVIMIENTOS_CSV, index=False)
    st.cache_data.clear()


def stock_actual_con_movimientos(df_inv: pd.DataFrame, df_catalogo: pd.DataFrame) -> pd.Series:
    """Último stock histórico (≥0) + ingresos manuales registrados desde entonces."""
    catalogo = df_catalogo.set_index("codigo_articulo")
    reales = catalogo.index[catalogo["es_producto"].astype(bool)]
    real = df_inv[df_inv["codigo_articulo"].isin(reales)].copy()
    ultimo = (real.sort_values("semana_del_año")
                  .groupby("codigo_articulo")["stock_fin_semana"].last()
                  .clip(lower=0))
    df_mov = cargar_movimientos_inventario()
    if len(df_mov):
        extra = df_mov.groupby("codigo_articulo")["cantidad"].sum()
        ultimo = ultimo.add(extra, fill_value=0)
    return ultimo


def texto_situacion_stock(dias) -> str:
    if dias <= 0:
        return "🔴 Ya no queda stock"
    if not np.isfinite(dias):
        return "⚪ Sin consumo reciente registrado"
    if dias <= 3:
        return f"🔴 ¡Urgente! Se acaba en {dias:.0f} día(s)"
    if dias <= 14:
        return f"🟠 Se acaba en {dias:.0f} días"
    return f"🟢 Alcanza para {dias:.0f}+ días"


@st.cache_data
def calcular_cobertura_por_familias(df_inv: pd.DataFrame, df_catalogo: pd.DataFrame,
                                     familias: int) -> pd.DataFrame:
    """Días de cobertura de stock por producto real, dado un número de familias.
    Sube el número de familias → sube el consumo esperado → bajan los días de
    cobertura, siempre y de forma verificable (regla de 3, no un modelo)."""
    catalogo = df_catalogo.set_index("codigo_articulo")
    catalogo = catalogo[catalogo["es_producto"].astype(bool)]
    tasas = calcular_tasa_percapita(df_inv, df_catalogo)
    stock = stock_actual_con_movimientos(df_inv, df_catalogo)

    filas = []
    for codigo, st_actual in stock.items():
        if codigo not in catalogo.index:
            continue
        info = catalogo.loc[codigo]
        tasa = tasas.get(codigo, 0.0)
        consumo_semana = tasa * familias
        consumo_dia = consumo_semana / 7.0
        if st_actual <= 0:
            dias = 0.0
        elif consumo_dia <= 1e-9:
            dias = np.inf
        else:
            dias = st_actual / consumo_dia
        filas.append({
            "Código": codigo, "Producto": info["nombre_producto"],
            "Categoría": info["categoria_general"], "Unidad": info["unidad_medida"],
            "Stock actual": round(float(st_actual), 1),
            "Consumo esperado/semana": round(float(consumo_semana), 2),
            "_dias": dias, "Situación": texto_situacion_stock(dias),
        })
    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return tabla
    return tabla.sort_values("_dias").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3ter. ALTA / EDICIÓN DE PACIENTES  (nuevo, feedback de Jairo 6 jul)
#    Persiste directamente sobre el CSV de pacientes (misma fuente que usa
#    cargar_datos() en modo CSV). El score del modelo nunca se expone al
#    personal: solo se traduce a una de estas 3 frases.
# ══════════════════════════════════════════════════════════════════════════════
def descripcion_prioridad(banda: str) -> str:
    return {
        "Alto": "🔴 Necesita atención pronto",
        "Revisión": "🟡 Revisar cuando se pueda",
        "Bajo": "🟢 Sin urgencia por ahora",
    }.get(banda, banda)


def señales_paciente(p) -> str:
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
    return ", ".join(señales[:3]) if señales else "sin señales de alarma específicas"


def cargar_pacientes_actual() -> pd.DataFrame:
    """Relee el CSV de pacientes en caliente (para reflejar altas/ediciones recién guardadas)."""
    return pd.read_csv(PACIENTES_CSV)


def guardar_paciente(datos: dict, id_existente=None) -> str:
    """Crea un paciente nuevo o actualiza uno existente en el CSV. Devuelve el id usado."""
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
                                 f"consumo ~{r['Consumo semanal']:g} {r['Unidad']}/semana")
                    if len(urgentes) > 6:
                        st.caption(f"… y {len(urgentes) - 6} más en la pestaña Inventario.")

        with col_b:
            with st.container(border=True):
                st.subheader("Pacientes con prioridad alta")
                if pac_alto.empty:
                    st.success("Sin pacientes clasificados con prioridad alta.")
                else:
                    for _, p in pac_alto.head(6).iterrows():
                        st.error(f"**{p['id_paciente']}** · {int(p['edad'])} años · "
                                 f"{p['diagnostico']} — señales: {señales_paciente(p)}")
                    if len(pac_alto) > 6:
                        st.caption(f"… y {len(pac_alto) - 6} más en la pestaña Pacientes.")

    # ── INVENTARIO ────────────────────────────────────────────────────────────
    with tab_inv:
        st.header("Plan de reposición — próximos 14 días")

        familias = st.number_input(
            "¿Cuántas familias hay hoy en el albergue?",
            min_value=1, max_value=200, value=ctx["ocupacion_actual"], step=1,
            help="El consumo esperado y los días de cobertura de abajo se "
                 "recalculan solos con este número.")
        cobertura = calcular_cobertura_por_familias(ctx["df_inv"], ctx["df_catalogo"], familias)

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

        vista = vista.drop(columns=["Stock actual", "Riesgo 7 días", "Riesgo 14 días",
                                    "Consumo semanal"]).merge(
            cobertura[["Código", "Stock actual", "Consumo esperado/semana", "Situación"]],
            on="Código", how="left")
        vista["Situación"] = vista["Situación"].fillna("⚪ Sin datos suficientes")
        vista = vista[["Producto", "Categoría", "Unidad", "Stock actual",
                       "Consumo esperado/semana", "Situación", "Estado", "Código"]]

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
                })
            st.caption(f"{len(agotados)} agotados · {len(criticos)} críticos · "
                       f"{len(planificar)} para planificar · calculado con **{familias} familias**.")

        st.divider()
        st.subheader("Registrar llegada de productos")
        with st.form("form_ingreso_inventario"):
            opciones = dict(zip(cobertura["Producto"], cobertura["Código"]))
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
                codigo_ingreso = opciones[producto_sel]
                guardar_ingreso_inventario(codigo_ingreso, cantidad_ingreso, nota_ingreso)
                st.success(f"Registrado: +{cantidad_ingreso:g} de **{producto_sel}**. "
                           "Cambia el número de familias o vuelve a esta pestaña para "
                           "ver la tabla actualizada.")

        with st.expander("Consultar un producto con otro número de familias"):
            st.caption("Para probar un escenario distinto sin cambiar el número de arriba.")
            c1, c2, c3 = st.columns([2, 1, 1], gap="medium")
            with c1:
                prod_consulta = st.selectbox("Producto", cobertura["Producto"].tolist(),
                                             key="consulta_producto")
            with c2:
                familias_hip = st.number_input("Familias a probar", 1, 200, familias, step=1,
                                               key="consulta_familias")
            with c3:
                consultar = st.button("Consultar", use_container_width=True)
            if consultar:
                cobertura_hip = calcular_cobertura_por_familias(
                    ctx["df_inv"], ctx["df_catalogo"], familias_hip)
                fila = cobertura_hip[cobertura_hip["Producto"] == prod_consulta].iloc[0]
                st.info(f"Con **{familias_hip} familias**: {fila['Situación']} · "
                        f"consumo esperado: {fila['Consumo esperado/semana']:g} "
                        f"{fila['Unidad']}/semana · stock actual: {fila['Stock actual']:g} "
                        f"{fila['Unidad']}.")

        with st.expander("Ver movimientos de inventario registrados a mano"):
            df_mov_ver = cargar_movimientos_inventario()
            if df_mov_ver.empty:
                st.caption("Todavía no se registró ningún ingreso manual.")
            else:
                st.dataframe(df_mov_ver.sort_values("fecha", ascending=False),
                            use_container_width=True, hide_index=True)

    # ── PACIENTES ─────────────────────────────────────────────────────────────
    with tab_pac:
        st.header("Pacientes que requieren atención primero")
        if pac_alto.empty:
            st.success("Ningún paciente en prioridad alta ahora mismo.")
        else:
            tabla = pac_alto[["id_paciente", "edad", "diagnostico", "etapa_cancer",
                              "estado_nutricional", "presencia_infeccion",
                              "adherencia_tratamiento"]].copy()
            tabla["presencia_infeccion"] = tabla["presencia_infeccion"].map({1: "Sí", 0: "No"})
            tabla.columns = ["Paciente", "Edad", "Diagnóstico", "Etapa",
                             "Nutrición", "Infección", "Sigue tratamiento"]
            st.dataframe(tabla, use_container_width=True, hide_index=True, height=280)
            st.caption(f"{len(pac_alto)} de {len(df_pac)} pacientes en prioridad alta. "
                       "Confirmar cada caso con el equipo médico.")

        st.divider()
        st.subheader("Situación general de los pacientes")
        for banda, etiqueta in [("Alto", "Prioridad alta"), ("Revisión", "En revisión"),
                                ("Bajo", "Prioridad baja")]:
            cnt = int((df_pac["banda_riesgo"] == banda).sum())
            pct = cnt / len(df_pac)
            st.progress(pct, text=f"{etiqueta}: {cnt} pacientes ({pct*100:.0f}%)")

        st.divider()
        # Nota: se usa st.radio() en vez de un segundo st.tabs() anidado dentro de
        # la pestaña "Pacientes" — se detectó en pruebas manuales que Streamlit
        # puede mezclar visualmente el contenido de pestañas anidadas después de
        # interactuar con un formulario. El radio evita ese problema por completo.
        accion_paciente = st.radio(
            "¿Qué quieres hacer?", ["➕ Registrar nuevo paciente", "✏️ Editar paciente existente"],
            horizontal=True, label_visibility="collapsed")
        if accion_paciente == "➕ Registrar nuevo paciente":
            formulario_paciente(ctx)
        else:
            editar_paciente_existente(ctx)


def formulario_paciente(ctx: dict) -> None:
    """Evalúa la prioridad de un paciente y, si se marca la casilla, lo registra
    como nuevo ingreso (el score nunca se muestra al personal)."""
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
        registrar = st.checkbox("Registrar este paciente como nuevo ingreso al guardar")
        enviado = st.form_submit_button("Evaluar prioridad de atención",
                                        use_container_width=True)

    if not enviado:
        return

    datos = {
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
    }
    X_new = pd.DataFrame([datos])

    # --- Score binario interno (Alto vs Bajo); nunca se muestra al personal ---
    score = score_paciente_alto(m_pac, X_new)[0]
    banda = banda_desde_score(score)

    if banda == "Alto":
        st.error("**PRIORIDAD ALTA** — Atender primero: avisar al médico tratante "
                 "y hacer seguimiento diario.")
    elif banda == "Revisión":
        st.warning("**ZONA DE REVISIÓN** — Caso ambiguo: se recomienda evaluación "
                   "clínica adicional antes de definir prioridad.")
    else:
        st.success("**PRIORIDAD BAJA** — Continuar con los controles habituales "
                   "según cronograma.")
    st.caption("Resultado orientativo; la evaluación clínica la confirma el personal de salud.")

    if registrar:
        datos_guardar = dict(datos)
        datos_guardar["nivel_riesgo"] = "Medio"  # etiqueta original: referencia, no la usa el modelo
        nuevo_id = guardar_paciente(datos_guardar)
        st.success(f"Registrado como nuevo paciente: **{nuevo_id}**. Aparecerá en las "
                   "listas de arriba la próxima vez que se actualice esta pantalla.")


def editar_paciente_existente(ctx: dict) -> None:
    """Carga un paciente ya registrado, permite ajustar sus variables accionables
    (no las clínicas) y guarda los cambios. Muestra el cambio de prioridad en
    lenguaje llano, nunca el score."""
    m_pac = ctx["m_pac"]
    df_pac_actual = cargar_pacientes_actual()
    if df_pac_actual.empty:
        st.info("No hay pacientes registrados todavía.")
        return

    id_sel = st.selectbox("Selecciona un paciente", df_pac_actual[PACIENTE_ID_COL].tolist(),
                          key="sel_editar_paciente")
    base = df_pac_actual[df_pac_actual[PACIENTE_ID_COL] == id_sel].iloc[0]
    fila_actual = df_pac_actual[df_pac_actual[PACIENTE_ID_COL] == id_sel]
    cols_modelo = columnas_paciente_modelo(fila_actual, m_pac)
    banda_antes = banda_desde_score(score_paciente_alto(m_pac, fila_actual[cols_modelo])[0])

    st.caption(f"Datos clínicos de referencia (no editables aquí): {int(base['edad'])} años · "
               f"{base['diagnostico']} · etapa {base['etapa_cancer']} · "
               f"{base['meses_en_tratamiento']} meses en tratamiento.")

    opciones_apoyo = ["Fuerte", "Moderado", "Limitado"]
    opciones_adherencia = ["Alta", "Media", "Baja"]
    opciones_acceso = ["Completo", "Parcial", "Limitado"]
    opciones_nutricion = ["Normal", "Desnutrición leve", "Desnutrición severa"]
    opciones_emocional = ["Estable", "Ansioso", "Deprimido"]

    with st.form(f"form_editar_{id_sel}"):
        e1, e2, e3 = st.columns(3)
        with e1:
            apoyo = st.selectbox("Apoyo familiar", opciones_apoyo,
                                 index=opciones_apoyo.index(base["apoyo_familiar"]))
            adherencia = st.selectbox("¿Sigue el tratamiento?", opciones_adherencia,
                                      index=opciones_adherencia.index(base["adherencia_tratamiento"]))
        with e2:
            acceso = st.selectbox("Acceso a medicamentos", opciones_acceso,
                                  index=opciones_acceso.index(base["acceso_medicamentos"]))
            nutricion = st.selectbox("Estado nutricional", opciones_nutricion,
                                     index=opciones_nutricion.index(base["estado_nutricional"]))
        with e3:
            emocional = st.selectbox("Estado emocional", opciones_emocional,
                                     index=opciones_emocional.index(base["estado_emocional_paciente"]))
        guardar_btn = st.form_submit_button("Guardar cambios", use_container_width=True)

    if not guardar_btn:
        return

    datos = dict(base)
    datos.update({"apoyo_familiar": apoyo, "adherencia_tratamiento": adherencia,
                  "acceso_medicamentos": acceso, "estado_nutricional": nutricion,
                  "estado_emocional_paciente": emocional})
    datos.pop(PACIENTE_ID_COL, None)
    guardar_paciente(datos, id_existente=id_sel)

    fila_nueva = pd.DataFrame([{**datos, PACIENTE_ID_COL: id_sel}])
    cols_modelo2 = columnas_paciente_modelo(fila_nueva, m_pac)
    banda_despues = banda_desde_score(score_paciente_alto(m_pac, fila_nueva[cols_modelo2])[0])

    if banda_antes != banda_despues:
        st.success(f"Guardado. Prioridad: {descripcion_prioridad(banda_antes)} → "
                   f"**{descripcion_prioridad(banda_despues)}**")
    else:
        st.success(f"Guardado. Prioridad: {descripcion_prioridad(banda_despues)}")


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

                with st.expander("Comparar con etiqueta original del dataset (3 clases)"):
                    st.caption(
                        "La columna `nivel_riesgo` proviene del dataset sintético original (3 clases) "
                        "y se muestra únicamente como referencia / comparación. El sistema en producción "
                        "clasifica y prioriza usando el score binario del modelo, no esta etiqueta. "
                        "(Sección movida aquí desde la vista de personal — feedback de Jairo, 6 jul.)"
                    )
                    tabla_ref = df_pac_modelo[["id_paciente", "nivel_riesgo", "banda_riesgo", "risk_score"]].copy()
                    tabla_ref.columns = ["Paciente", "Etiqueta original (CSV)", "Banda del modelo", "Score"]
                    st.dataframe(tabla_ref, use_container_width=True, hide_index=True, height=240)

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
