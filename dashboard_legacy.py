"""
⚠️ ARCHIVO LEGACY — NO ES EL DASHBOARD VIGENTE ⚠️
──────────────────────────────────────────────────
Esta es la versión ANTERIOR del dashboard (previa al rediseño "full UI/UX" de
julio). Se conserva solo como referencia/backup.

- La app que corre en producción es `dashboard.py` (raíz del proyecto).
- Este archivo todavía carga `models/models_pacientes.pkl`, el modelo de
  pacientes MULTICLASE (Alto/Medio/Bajo) que ya fue reemplazado por el modelo
  binario `models/risk_model_rf_binary.joblib` (ver `dashboard.py`).
- No usar este archivo para la demo, el video ni la sustentación.

Si el equipo confirma que ya no hace falta, se puede eliminar o mover a una
carpeta `archive/`; mientras tanto queda marcado aquí para evitar confusiones
al revisar el repo. Ver `revision_pendientes_hito4.md`, sección 4.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib, warnings, os
from datetime import datetime
warnings.filterwarnings('ignore')

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ALDIMI Predict",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE = os.getcwd()

# ── PALETA ────────────────────────────────────────────────────────────────────
AZUL    = "#1b3b64"
CELESTE = "#4da9d4"
DORADO  = "#c6a356"
BLANCO  = "#ffffff"
FONDO   = "#f8f9fa"
GRIS    = "#333333"
GRID    = "#dee2e6"
ROJO    = "#c0392b"
NARANJA = "#e67e22"
VERDE   = "#27ae60"

# ── HELPERS ───────────────────────────────────────────────────────────────────
def CL(height=320, margin=None, show_legend=False):
    m = margin or dict(l=4, r=4, t=24, b=4)
    return dict(height=height, plot_bgcolor='rgba(248,249,250,0.9)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Segoe UI', size=12, color=GRIS),
                margin=m, showlegend=show_legend)

def ax(grid=True, title='', rng=None):
    d = dict(title=dict(text=title, font=dict(color=GRIS, size=11)),
             tickfont=dict(color=GRIS, size=11),
             showgrid=grid, gridcolor=GRID, zeroline=False,
             linecolor=GRID, showline=True)
    if rng: d['range'] = rng
    return d

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
.main {{ background-color:{FONDO}; }}
.block-container {{ padding-top:1.1rem; padding-bottom:1.4rem; max-width:1320px; }}
html, body, [class*="css"] {{ font-family:'Segoe UI',sans-serif; color:{GRIS}; }}

section[data-testid="stSidebar"] {{ background-color:{AZUL} !important; }}
section[data-testid="stSidebar"] * {{ color:{BLANCO} !important; }}
section[data-testid="stSidebar"] .stRadio label {{
  background:rgba(255,255,255,0.08); border-radius:8px;
  padding:8px 12px; margin-bottom:4px; display:block; transition:background 0.2s;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{ background:rgba(255,255,255,0.20); }}
section[data-testid="stSidebar"] hr {{ border-color:rgba(255,255,255,0.18); }}

.kpi {{ background:{BLANCO}; border-radius:10px; padding:14px 16px;
        box-shadow:0 1px 6px rgba(0,0,0,0.09); border-top:4px solid {CELESTE}; text-align:center; }}
.kpi.danger {{ border-top-color:{ROJO}; }}
.kpi.warn   {{ border-top-color:{DORADO}; }}
.kpi.ok     {{ border-top-color:{VERDE}; }}
.kpi-num    {{ font-size:1.85rem; font-weight:700; color:{AZUL}; margin:0 0 2px 0; }}
.kpi-label  {{ font-size:0.74rem; color:#666; text-transform:uppercase; letter-spacing:0.05em; margin:0; }}
.kpi-sub    {{ font-size:0.70rem; color:#999; margin:3px 0 0 0; }}

.ph {{ background:{AZUL}; color:white; padding:13px 20px; border-radius:10px; margin-bottom:16px; }}
.ph h1 {{ margin:0; font-size:1.35rem; color:white; font-weight:700; }}
.ph p  {{ margin:3px 0 0; font-size:0.82rem; color:rgba(255,255,255,0.72); }}

.st {{ font-size:0.95rem; font-weight:700; color:{AZUL};
       border-left:3px solid {CELESTE}; padding-left:9px; margin:0 0 10px 0; }}

.card {{ background:{BLANCO}; border-radius:10px; padding:14px 16px;
         box-shadow:0 1px 6px rgba(0,0,0,0.08); margin-bottom:12px; }}

.r-alto   {{ background:#fdf3f2; border:1.5px solid {ROJO};   border-radius:8px; padding:12px 16px; margin-bottom:10px; }}
.r-medio  {{ background:#fef9f0; border:1.5px solid {DORADO}; border-radius:8px; padding:12px 16px; margin-bottom:10px; }}
.r-bajo   {{ background:#f0faf4; border:1.5px solid {VERDE};  border-radius:8px; padding:12px 16px; margin-bottom:10px; }}
.r-alerta {{ background:#fdf3f2; border:1.5px solid {ROJO};   border-radius:8px; padding:12px 16px; margin-bottom:10px; }}
.r-ok     {{ background:#f0faf4; border:1.5px solid {VERDE};  border-radius:8px; padding:12px 16px; margin-bottom:10px; }}
.r-tit    {{ font-size:1.05rem; font-weight:700; margin:0 0 5px 0; }}
.r-acc    {{ font-size:0.86rem; margin:0; color:{GRIS}; line-height:1.5; }}

.accion {{ background:{BLANCO}; border-left:4px solid {ROJO}; border-radius:6px;
           padding:8px 12px; margin-bottom:6px; font-size:0.86rem;
           box-shadow:0 1px 4px rgba(0,0,0,0.06); }}
.accion.warn {{ border-left-color:{DORADO}; }}
.accion b {{ color:{AZUL}; }}

div.stButton>button, div.stFormSubmitButton>button, div.stDownloadButton>button {{
  background-color:{CELESTE}; color:white; border:none; border-radius:8px;
  padding:9px 22px; font-weight:600; font-size:0.92rem;
  width:100%; transition:background 0.18s;
}}
div.stButton>button:hover, div.stFormSubmitButton>button:hover,
div.stDownloadButton>button:hover {{ background-color:{AZUL}; color:white; }}

[data-baseweb="slider"] [role="slider"] {{ background-color:{CELESTE} !important; border-color:{CELESTE} !important; }}
div[data-baseweb="slider"] > div > div > div:nth-child(1) > div {{ background:{CELESTE} !important; }}
div[data-testid="stForm"] {{ border:none !important; padding:0 !important; background:transparent !important; box-shadow:none !important; }}
[data-testid="stDataFrame"] {{ border-radius:8px; overflow:hidden; }}
</style>
""", unsafe_allow_html=True)


# ── CARGA ─────────────────────────────────────────────────────────────────────
DB_COMUN = os.path.join(BASE, 'data', 'aldimi_core.db')

@st.cache_data
def cargar_datos():
    """Lee de la BD común (confluencia con el módulo de IA); si no existe, usa los CSV."""
    if os.path.exists(DB_COMUN):
        import sqlite3
        con = sqlite3.connect(DB_COMUN)
        try:
            tablas = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            if {'inventario_semanal', 'articulos', 'pacientes',
                'catalogo_productos'} <= tablas:
                df_inv  = pd.read_sql('SELECT * FROM inventario_semanal', con)
                df_orig = pd.read_sql('SELECT * FROM articulos', con)
                df_pac  = pd.read_sql('SELECT * FROM pacientes', con)
                df_cat  = pd.read_sql('SELECT * FROM catalogo_productos', con)
                if 'nombre_producto' in df_inv.columns:
                    return df_inv, df_orig, df_pac, df_cat, 'BD común (aldimi_core.db)'
            # BD desactualizada (sin catálogo): usar CSV y avisar
        finally:
            con.close()
    df_inv  = pd.read_csv(os.path.join(BASE, 'data', 'processed', 'aldimi_dataset_semanal.csv'))
    df_orig = pd.read_csv(os.path.join(BASE, 'data', 'processed', 'aldimi_dataset_completo.csv'))
    df_pac  = pd.read_csv(os.path.join(BASE, 'data', 'processed', 'aldimi_pacientes_sintetico.csv'))
    df_cat  = pd.read_csv(os.path.join(BASE, 'data', 'processed', 'catalogo_productos.csv'))
    return df_inv, df_orig, df_pac, df_cat, 'Archivos CSV locales (ejecuta integracion_bd.py para actualizar la BD)'

@st.cache_resource
def cargar_modelos():
    m_inv = joblib.load(os.path.join(BASE, 'models', 'models_inventario.pkl'))
    m_pac = joblib.load(os.path.join(BASE, 'models', 'models_pacientes.pkl'))
    return m_inv, m_pac

try:
    df_inv, df_orig, df_pac, df_catalogo, fuente_datos = cargar_datos()
except FileNotFoundError as e:
    st.error(f"❌ Archivos de datos no encontrados: {e}"); st.stop()

# El catálogo separa el código interno (identificador en la base) del nombre
# visible del producto, y añade unidad de medida estándar y categoría general.
df_catalogo['es_producto'] = df_catalogo['es_producto'].astype(bool)
CATALOGO = df_catalogo.set_index('codigo_articulo')
# unidad de medida estándar por tipo de producto (categoría del almacén)
UNIDAD_POR_TIPO = (df_catalogo.groupby('categoria')['unidad_medida']
                              .agg(lambda s: s.mode().iat[0]).to_dict())
try:
    m_inv, m_pac = cargar_modelos()
except FileNotFoundError as e:
    st.error(f"❌ Modelos no encontrados: {e}"); st.stop()


# ── PLAN DE REPOSICIÓN (predicción sobre la última semana de cada producto) ──
# La app muestra SOLO productos reales del almacén (con nombre y unidad).
# Los registros técnicos/sintéticos se usan para entrenar los modelos,
# pero no aparecen en la vista operativa.
@st.cache_data
def plan_reposicion():
    ult = (df_inv.sort_values('semana_del_año')
                 .groupby('codigo_articulo', as_index=False).tail(1).copy())
    ult = ult[ult['codigo_articulo'].isin(
        CATALOGO.index[CATALOGO['es_producto']])].copy()
    cat_enc = m_inv['le_categoria'].transform(ult['categoria'])
    X = np.column_stack([
        cat_enc,
        ult['ocupacion_albergue'],
        ult['stock_fin_semana'],                # stock con el que inicia la próxima semana
        np.zeros(len(ult)),                     # sin ingresos confirmados
        ult['rolling_avg_salidas_3sem'],        # consumo esperado
        ult['rolling_avg_salidas_3sem'],
        np.minimum(ult['semana_del_año'] + 1, 52),
    ])
    X_sc = m_inv['scaler'].transform(X)
    prob7  = m_inv['modelo_7d'].predict_proba(X_sc)[:, 1]
    prob14 = m_inv['modelo_14d'].predict_proba(X_sc)[:, 1]

    stock = ult['stock_fin_semana'].values
    info  = CATALOGO.loc[ult['codigo_articulo']]
    plan = pd.DataFrame({
        'Producto':          info['nombre_producto'].values,
        'Categoría':         info['categoria_general'].values,
        'Unidad':            info['unidad_medida'].values,
        'Stock actual':      np.maximum(ult['stock_fin_semana'].round(1).values, 0),
        'Consumo semanal':   ult['rolling_avg_salidas_3sem'].round(1).values,
        'Riesgo 7 días':     (prob7 * 100).round(0).astype(int),
        'Riesgo 14 días':    (prob14 * 100).round(0).astype(int),
        'Código':            ult['codigo_articulo'].values,   # apoyo de búsqueda
    })
    plan['Acción recomendada'] = np.select(
        [stock <= 0, prob7 >= 0.5, prob14 >= 0.5],
        ['🔴 Agotado — reponer ya', '🟠 Se agota esta semana', '🟡 Planificar compra / donación'],
        default='🟢 Cubierto')
    orden = plan['Acción recomendada'].map(
        {'🔴 Agotado — reponer ya': 0, '🟠 Se agota esta semana': 1,
         '🟡 Planificar compra / donación': 2, '🟢 Cubierto': 3})
    # dentro de cada grupo, primero lo que más se consume (mayor impacto en la olla)
    return (plan.assign(_o=orden)
                .sort_values(['_o', 'Consumo semanal'], ascending=[True, False])
                .drop(columns='_o'))

plan = plan_reposicion()
agotados    = plan[plan['Acción recomendada'].str.startswith('🔴')]
urgentes    = plan[plan['Acción recomendada'].str.startswith(('🔴', '🟠'))]
planificar  = plan[plan['Acción recomendada'].str.startswith('🟡')]
pac_alto    = df_pac[df_pac['nivel_riesgo'] == 'Alto']
total_pac   = len(df_pac)
ocupacion_actual = int(df_inv.loc[df_inv['semana_del_año'].idxmax(), 'ocupacion_albergue'])
semana_actual    = int(df_inv['semana_del_año'].max())

metricas_inv = m_inv.get('metricas_7d', {})
metricas_pac = m_pac.get('metricas', {})
nombre_mod_inv = m_inv.get('mejor_nombre_7d', '')
nombre_mod_pac = m_pac.get('mejor_nombre', '')


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:8px 0 4px'>
      <span style='font-size:2rem'>🏥</span>
      <p style='font-weight:700;font-size:1.05rem;margin:4px 0 0'>ALDIMI Predict</p>
      <p style='font-size:0.78rem;opacity:0.75;margin:2px 0 0'>Albergue Divina Misericordia</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    pagina = st.radio("Navegación", [
        "🏠  Inicio",
        "📦  Inventario y reposición",
        "👤  Pacientes y prioridad",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.82rem;line-height:2'>
      <p style='font-weight:700;margin:0 0 4px'>📋 Resumen de hoy</p>
      🔴 <b>{len(agotados)}</b> productos agotados<br>
      🟠 <b>{len(urgentes) - len(agotados)}</b> se agotan esta semana<br>
      👤 <b>{len(pac_alto)}</b> pacientes con prioridad alta<br>
      🏠 <b>{ocupacion_actual}</b> familias alojadas
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    with st.expander("ℹ️ Acerca del sistema"):
        f1_inv = metricas_inv.get(nombre_mod_inv, {}).get('f1', 0)
        f1_pac = metricas_pac.get(nombre_mod_pac, {}).get('f1_macro', 0)
        st.markdown(f"""
        <p style='font-size:0.76rem;line-height:1.7;opacity:0.9'>
        Las alertas se calculan con modelos de inteligencia artificial entrenados
        con el historial del albergue.<br><br>
        · Alertas de stock: acierta <b>{f1_inv*100:.0f} de cada 100</b> casos.<br>
        · Prioridad de pacientes: acierta <b>3 de cada 4</b> evaluaciones.<br><br>
        Las recomendaciones apoyan la decisión del equipo;
        no reemplazan el criterio del personal de salud.<br><br>
        <span style='opacity:0.7'>Modelos: {nombre_mod_inv} (inventario) ·
        {nombre_mod_pac} (pacientes)<br>
        Fuente de datos: {fuente_datos}</span>
        </p>""", unsafe_allow_html=True)
    st.markdown(f"""
    <p style='font-size:0.72rem;opacity:0.55;margin-top:8px'>
      ML 1ACC0057 · UPC · Julio 2026
    </p>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# INICIO — PANEL DE GESTIÓN
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "🏠  Inicio":

    st.markdown(f"""<div class='ph'>
      <h1>🏥 Panel de gestión del albergue</h1>
      <p>Semana {semana_actual} del año · {ocupacion_actual} familias alojadas ·
         Lo importante primero: qué reponer y a quién atender.</p>
    </div>""", unsafe_allow_html=True)

    # ── KPIs accionables ─────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="kpi danger">
          <p class="kpi-num">{len(agotados)}</p>
          <p class="kpi-label">🔴 Agotados — reponer ya</p>
          <p class="kpi-sub">Productos sin stock en el almacén</p>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi warn">
          <p class="kpi-num">{len(urgentes) - len(agotados) + len(planificar)}</p>
          <p class="kpi-label">🟠 Por agotarse</p>
          <p class="kpi-sub">Se acabarían en los próximos 7–14 días</p>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi danger">
          <p class="kpi-num">{len(pac_alto)}</p>
          <p class="kpi-label">👤 Prioridad alta</p>
          <p class="kpi-sub">Pacientes que requieren atención primero</p>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi ok">
          <p class="kpi-num">{ocupacion_actual}</p>
          <p class="kpi-label">🏠 Familias alojadas</p>
          <p class="kpi-sub">Capacidad ALDIMI 2.0: 100 familias</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── Acciones de la semana ────────────────────────────────────────────────
    col_a, col_b = st.columns(2, gap="medium")

    with col_a:
        st.markdown('<p class="st">📦 Reposición urgente — esta semana</p>', unsafe_allow_html=True)
        if len(urgentes) == 0:
            st.markdown(f"""<div class="r-ok"><p class="r-acc">
              ✅ Ningún producto en riesgo de agotarse esta semana.</p></div>""",
              unsafe_allow_html=True)
        else:
            st.caption("Los de mayor consumo primero — son los que más impactan en la cocina y la despensa.")
            for _, r in urgentes.head(6).iterrows():
                detalle = ('sin stock' if r['Acción recomendada'].startswith('🔴')
                           else f"quedan {r['Stock actual']:g} {r['Unidad']}")
                st.markdown(f"""<div class="accion">
                  <b>{r['Producto']}</b> · {r['Categoría']} — {detalle},
                  se consumen ~{r['Consumo semanal']:g} {r['Unidad']}/semana
                  <span style="float:right;color:{ROJO};font-weight:700">{r['Riesgo 7 días']}%</span>
                </div>""", unsafe_allow_html=True)
            if len(urgentes) > 6:
                st.caption(f"… y {len(urgentes)-6} productos más. "
                           "Ver la lista completa en **📦 Inventario y reposición**.")

    with col_b:
        st.markdown('<p class="st">👤 Pacientes con prioridad alta</p>', unsafe_allow_html=True)
        if len(pac_alto) == 0:
            st.markdown(f"""<div class="r-ok"><p class="r-acc">
              ✅ Ningún paciente clasificado con prioridad alta.</p></div>""",
              unsafe_allow_html=True)
        else:
            muestra = pac_alto.head(6)
            for _, p in muestra.iterrows():
                senales = []
                if p['etapa_cancer'] in ('III', 'IV'):        senales.append(f"etapa {p['etapa_cancer']}")
                if p['presencia_infeccion'] == 1:             senales.append("infección")
                if p['estado_nutricional'] != 'Normal':       senales.append("desnutrición")
                if p['perdida_peso_reciente'] == 1:           senales.append("pérdida de peso")
                detalle = ', '.join(senales[:3]) if senales else 'revisar historia'
                st.markdown(f"""<div class="accion">
                  <b>{p['id_paciente']}</b> · {int(p['edad'])} años · {p['diagnostico']}
                  <span style="display:block;font-size:0.78rem;color:#888">Señales: {detalle}</span>
                </div>""", unsafe_allow_html=True)
            if len(pac_alto) > 6:
                st.caption(f"… y {len(pac_alto)-6} pacientes más. "
                           "Ver el detalle en **👤 Pacientes y prioridad**.")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Contexto: almacén y pacientes ────────────────────────────────────────
    col_c, col_d = st.columns(2, gap="medium")

    with col_c:
        st.markdown('<p class="st">📊 ¿Dónde está el riesgo del almacén?</p>', unsafe_allow_html=True)
        riesgo_cat = (plan.assign(urg=plan['Acción recomendada'].str.startswith(('🔴', '🟠')),
                                  pla=plan['Acción recomendada'].str.startswith('🟡'))
                          .groupby('Categoría')[['urg', 'pla']].sum()
                          .assign(total=lambda d: d.urg + d.pla)
                          .query('total > 0')
                          .sort_values('total').tail(10))
        if len(riesgo_cat):
            fig_rc = go.Figure()
            fig_rc.add_trace(go.Bar(y=riesgo_cat.index, x=riesgo_cat['urg'], name='Reponer ya',
                                    orientation='h', marker_color=ROJO))
            fig_rc.add_trace(go.Bar(y=riesgo_cat.index, x=riesgo_cat['pla'], name='Planificar',
                                    orientation='h', marker_color=DORADO))
            fig_rc.update_layout(**CL(300, dict(l=4, r=10, t=28, b=4), show_legend=True),
                                 barmode='stack',
                                 xaxis=ax(True, 'Productos en riesgo'),
                                 yaxis=ax(False, '') | dict(automargin=True),
                                 legend=dict(orientation='h', y=1.12, x=0,
                                             font=dict(size=10, color=GRIS)))
            st.plotly_chart(fig_rc, width='stretch')
        else:
            st.success("Sin categorías en riesgo esta semana.")

    with col_d:
        st.markdown('<p class="st">👥 Situación general de los pacientes</p>', unsafe_allow_html=True)
        vc = df_pac['nivel_riesgo'].value_counts().reindex(['Bajo', 'Medio', 'Alto'])
        fig_donut = go.Figure(go.Pie(
            labels=['Prioridad baja', 'Prioridad media', 'Prioridad alta'],
            values=vc.values, hole=0.55,
            marker=dict(colors=[VERDE, NARANJA, ROJO], line=dict(color=FONDO, width=3)),
            textinfo='label+percent', textfont=dict(size=11, color=GRIS), pull=[0, 0, 0.05],
        ))
        fig_donut.update_layout(
            **CL(300, dict(l=4, r=4, t=8, b=30), show_legend=False),
            annotations=[dict(text=f'<b>{total_pac}</b><br><span style="font-size:10px">pacientes</span>',
                              x=0.5, y=0.5, showarrow=False, font_size=15)],
        )
        st.plotly_chart(fig_donut, width='stretch')

    st.markdown(f"""<div style="background:{AZUL};color:white;border-radius:8px;padding:10px 18px;
                font-size:0.83rem;margin-top:6px;line-height:1.75">
      <strong>🎯 Nuestra misión con la tecnología</strong> &nbsp;·&nbsp;
      Que nunca falte comida en la mesa (<strong>ODS 2</strong>) &nbsp;·&nbsp;
      Que ningún niño espere la atención que necesita (<strong>ODS 3</strong>) &nbsp;·&nbsp;
      Herramienta gratuita para organizaciones sociales (<strong>ODS 10</strong>)
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# INVENTARIO Y REPOSICIÓN
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📦  Inventario y reposición":

    st.markdown(f"""<div class='ph'>
      <h1>📦 Inventario y reposición</h1>
      <p>Qué comprar o pedir en donación, y cuándo — antes de que falte.</p>
    </div>""", unsafe_allow_html=True)

    # ── Plan de reposición semanal ────────────────────────────────────────────
    st.markdown('<p class="st">🛒 Plan de reposición — próximos 14 días</p>', unsafe_allow_html=True)
    c_filtro, c_buscar, c_desc = st.columns([2, 2, 1], gap="medium")
    with c_filtro:
        filtro = st.radio("Mostrar", ["Solo con acción pendiente", "Todos los productos"],
                          horizontal=True, label_visibility="collapsed")
    with c_buscar:
        busqueda = st.text_input("Buscar producto",
                                 placeholder="Buscar por nombre o código…",
                                 label_visibility="collapsed")
    plan_vista = plan if filtro == "Todos los productos" else \
        plan[~plan['Acción recomendada'].str.startswith('🟢')]
    if busqueda.strip():
        q = busqueda.strip().lower()
        plan_vista = plan_vista[
            plan_vista['Producto'].str.lower().str.contains(q, regex=False) |
            plan_vista['Código'].str.lower().str.contains(q, regex=False)]
    with c_desc:
        st.download_button(
            "⬇️ Descargar lista (Excel/CSV)",
            plan_vista.to_csv(index=False).encode('utf-8-sig'),
            file_name=f"plan_reposicion_semana{semana_actual}.csv",
            mime="text/csv")

    if len(plan_vista) == 0:
        st.success("✅ Todo el almacén está cubierto para las próximas dos semanas."
                   if not busqueda.strip() else "No se encontraron productos con esa búsqueda.")
    else:
        st.dataframe(
            plan_vista, width='stretch', hide_index=True, height=330,
            column_config={
                'Producto': st.column_config.TextColumn('Producto', width='large'),
                'Unidad':   st.column_config.TextColumn('Unidad', help='Unidad de medida estándar del producto'),
                'Código':   st.column_config.TextColumn('Código', help='Código interno — solo como apoyo de búsqueda'),
                'Riesgo 7 días':  st.column_config.ProgressColumn(
                    'Riesgo 7 días', format='%d%%', min_value=0, max_value=100),
                'Riesgo 14 días': st.column_config.ProgressColumn(
                    'Riesgo 14 días', format='%d%%', min_value=0, max_value=100),
            })
        st.caption(f"🔴 {len(agotados)} agotados · 🟠 {len(urgentes) - len(agotados)} se agotan esta semana · "
                   f"🟡 {len(planificar)} para planificar · "
                   "El stock y el consumo están en la unidad de cada producto. "
                   "El riesgo es la probabilidad de quedarse sin stock estimada por el sistema. "
                   "El código interno se muestra solo como apoyo de búsqueda.")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Consulta individual ───────────────────────────────────────────────────
    st.markdown('<p class="st">🔍 ¿Alcanzará el stock? — consulta un producto</p>', unsafe_allow_html=True)
    st.caption("Útil para simular: ¿y si llega una donación? ¿y si suben las familias alojadas? "
               "Ingresa las cantidades en la unidad del producto (kg, litros, latas, paquetes o unidades).")
    c1, c2, c3 = st.columns([2, 2, 1], gap="medium")
    with c1:
        cat_sel      = st.selectbox("Tipo de producto", m_inv['categorias'])
        unidad_sel   = UNIDAD_POR_TIPO.get(cat_sel, 'unidades')
        stock_inicio = st.number_input(f"Stock disponible hoy ({unidad_sel})", -50.0, 500.0, 8.0, 1.0)
        ingresos     = st.number_input(f"Donaciones/compras que llegan esta semana ({unidad_sel})", 0.0, 200.0, 0.0, 1.0)
    with c2:
        salidas     = st.number_input(f"Consumo previsto esta semana ({unidad_sel})", 0.0, 100.0, 3.0, 0.5)
        rolling_avg = st.number_input(f"Consumo promedio de las últimas 3 semanas ({unidad_sel})", 0.0, 50.0, 2.5, 0.5)
        ocupacion   = st.slider("Familias en el albergue", 40, 100, ocupacion_actual)
    with c3:
        semana    = st.slider("Semana del año", 1, 52, semana_actual)
        st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
        consultar = st.button("Evaluar", type="primary")

    if consultar:
        cat_enc = m_inv['le_categoria'].transform([cat_sel])[0]
        X_new   = np.array([[cat_enc, ocupacion, stock_inicio, ingresos, salidas, rolling_avg, semana]])
        X_sc    = m_inv['scaler'].transform(X_new)
        pred_7  = m_inv['modelo_7d'].predict(X_sc)[0]
        prob_7  = m_inv['modelo_7d'].predict_proba(X_sc)[0][1]
        pred_14 = m_inv['modelo_14d'].predict(X_sc)[0]
        prob_14 = m_inv['modelo_14d'].predict_proba(X_sc)[0][1]

        st.markdown("---")
        r1, r2, r3 = st.columns([2, 2, 3], gap="medium")

        def tarjeta_inv(pred, prob, dias):
            css = 'r-alerta' if pred else 'r-ok'
            col = ROJO if pred else VERDE
            lbl = 'SE AGOTARÍA' if pred else 'Alcanza'
            ico = '⚠️' if pred else '✅'
            acc = ('Pedir reposición o donación esta misma semana.' if dias == 7
                   else 'Incluirlo en la próxima campaña de donaciones.') if pred else \
                  ('El stock cubre esta semana.' if dias == 7
                   else 'El stock cubre las próximas dos semanas.')
            st.markdown(f"""<div class="{css}">
              <p class="r-tit" style="color:{col}">{ico} En {dias} días — {lbl}</p>
              <p class="r-acc">Probabilidad de quedarse sin stock: <strong>{prob*100:.0f}%</strong></p>
              <p class="r-acc" style="margin-top:5px">→ {acc}</p>
            </div>""", unsafe_allow_html=True)

        with r1: tarjeta_inv(pred_7,  prob_7,  7)
        with r2: tarjeta_inv(pred_14, prob_14, 14)
        with r3:
            fig_g = make_subplots(rows=1, cols=2,
                                  specs=[[{'type': 'indicator'}, {'type': 'indicator'}]])
            for ci, (label, prob, pred) in enumerate(
                    [('7 días', prob_7, pred_7), ('14 días', prob_14, pred_14)], 1):
                bar_c = ROJO if pred else VERDE
                fig_g.add_trace(go.Indicator(
                    mode="gauge+number", value=round(prob * 100, 1),
                    number={'suffix': '%', 'font': {'size': 22, 'color': AZUL}},
                    title={'text': f"<b>{label}</b>", 'font': {'size': 11, 'color': GRIS}},
                    gauge={'axis': {'range': [0, 100], 'tickfont': {'size': 9}},
                           'bar': {'color': bar_c, 'thickness': 0.28},
                           'bgcolor': 'rgba(248,249,250,1)',
                           'steps': [{'range': [0, 40],  'color': '#d5f5e3'},
                                     {'range': [40, 70], 'color': '#fdebd0'},
                                     {'range': [70, 100],'color': '#fadbd8'}],
                           'threshold': {'line': {'color': AZUL, 'width': 2}, 'value': 50}}
                ), row=1, col=ci)
            fig_g.update_layout(height=190, paper_bgcolor='rgba(0,0,0,0)',
                                margin=dict(l=10, r=10, t=28, b=8),
                                font=dict(family='Segoe UI', color=GRIS))
            st.plotly_chart(fig_g, width='stretch')

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Evolución + artículos de mayor consumo ────────────────────────────────
    col_ev, col_top = st.columns([3, 2], gap="medium")

    with col_ev:
        st.markdown('<p class="st">📈 ¿Cómo se mueve el stock durante el año?</p>', unsafe_allow_html=True)
        cat_sel2 = st.selectbox("Categoría general", sorted(df_inv['categoria_general'].dropna().unique()), key='cat2')
        df_cat = (df_inv[df_inv['categoria_general'] == cat_sel2]
                  .groupby('semana_del_año')
                  .agg(stock_promedio=('stock_fin_semana', 'mean'),
                       alertas_activas=('alerta_7_dias', 'sum'))
                  .reset_index())
        fig_ev = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ev.add_trace(go.Scatter(
            x=df_cat['semana_del_año'], y=df_cat['stock_promedio'],
            name='Stock promedio', line=dict(color=CELESTE, width=2.5),
            fill='tozeroy', fillcolor='rgba(77,169,212,0.15)',
            mode='lines+markers', marker=dict(size=4, color=CELESTE),
        ), secondary_y=False)
        fig_ev.add_trace(go.Bar(
            x=df_cat['semana_del_año'], y=df_cat['alertas_activas'],
            name='Semanas con faltantes', marker_color='rgba(192,57,43,0.40)',
        ), secondary_y=True)
        fig_ev.add_hline(y=0, line_dash='dash', line_color=ROJO, line_width=1.2,
                         annotation_text='Stock 0', annotation_font_color=ROJO, annotation_font_size=9)
        fig_ev.update_layout(**CL(310, dict(l=4, r=4, t=32, b=4), show_legend=True),
                             xaxis=ax(True, 'Semana del año'),
                             legend=dict(orientation='h', y=1.1, x=0, font=dict(size=10, color=GRIS)))
        fig_ev.update_yaxes(title_text="Stock (uds)", secondary_y=False,
                            showgrid=True, gridcolor=GRID, tickfont=dict(color=GRIS),
                            title_font=dict(color=GRIS))
        fig_ev.update_yaxes(title_text="Faltantes", secondary_y=True,
                            showgrid=False, tickfont=dict(color=GRIS), title_font=dict(color=GRIS))
        st.plotly_chart(fig_ev, width='stretch')
        st.caption("Sirve para anticipar los meses de mayor consumo y organizar campañas de donación a tiempo.")

    with col_top:
        st.markdown('<p class="st">⚡ Productos que más rápido se consumen</p>', unsafe_allow_html=True)
        top15 = (df_orig[df_orig['es_producto'].astype(bool)]
                 .nlargest(12, 'tasa_rotacion')[['codigo_articulo', 'nombre_producto', 'tasa_rotacion']]
                 .copy())
        top15['etiqueta'] = top15['nombre_producto'].str.slice(0, 28)
        cap = float(np.percentile(top15['tasa_rotacion'], 90)) * 1.15
        top15['valor_vis'] = top15['tasa_rotacion'].clip(upper=cap)
        top15['label']     = top15['tasa_rotacion'].apply(lambda v: f"{v:.2f} ⚡" if v > cap else f"{v:.2f}")
        colors = [ROJO if v > 1 else (DORADO if v > 0.5 else CELESTE) for v in top15['tasa_rotacion']]
        fig_top = go.Figure(go.Bar(
            x=top15['valor_vis'], y=top15['etiqueta'], orientation='h',
            marker_color=colors, text=top15['label'], textposition='outside',
            textfont=dict(size=10, color=GRIS), cliponaxis=False,
        ))
        fig_top.add_vline(x=min(1.0, cap), line_dash='dash', line_color=ROJO, line_width=1.3,
                          annotation_text='Crítico', annotation_font_color=ROJO,
                          annotation_font_size=9, annotation_position='top right')
        fig_top.update_layout(**CL(310, dict(l=4, r=52, t=10, b=4), show_legend=False),
                              xaxis=ax(True, 'Velocidad de consumo') | {'range': [0, cap * 1.12]},
                              yaxis=ax(False, '') | {'automargin': True, 'tickfont': dict(size=9, color=GRIS)})
        st.plotly_chart(fig_top, width='stretch')
        st.caption("Estos productos conviene pedirlos siempre en las campañas de donación.")

    with st.expander("🔬 Ficha técnica del modelo (para el equipo técnico)"):
        if metricas_inv:
            df_cmp = pd.DataFrame(metricas_inv).T
            df_cmp.columns = ['F1-Score', 'AUC-ROC']
            st.markdown(f"Modelo en producción a 7 días: **{nombre_mod_inv}** · "
                        f"a 14 días: **{m_inv.get('mejor_nombre_14d', '')}**. "
                        "Entrenados con validación cruzada estratificada (5 folds) y "
                        "búsqueda de hiperparámetros (RandomizedSearchCV).")
            st.dataframe(df_cmp.round(4), width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# PACIENTES Y PRIORIDAD
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "👤  Pacientes y prioridad":

    st.markdown(f"""<div class='ph'>
      <h1>👤 Pacientes y prioridad de atención</h1>
      <p>Evalúa a un paciente al ingresar y conoce quiénes necesitan atención primero.
         La decisión final siempre es del personal de salud.</p>
    </div>""", unsafe_allow_html=True)

    col_form, col_res = st.columns([3, 2], gap="medium")

    with col_form:
        with st.form("form_paciente", clear_on_submit=False):
            st.markdown('<p class="st">📋 Datos del paciente</p>', unsafe_allow_html=True)
            g1, g2, g3 = st.columns(3)
            with g1:
                edad       = st.number_input("Edad (años)", 2, 17, 8)
                sexo       = st.selectbox("Sexo", ["Masculino", "Femenino"])
                distancia  = st.number_input("Distancia a Lima (km)", 10, 1400, 600)
            with g2:
                lugar      = st.selectbox("Procedencia", ['Sierra sur', 'Sierra norte', 'Sierra centro',
                                                          'Selva', 'Costa norte', 'Costa sur', 'Lima'])
                instruccion = st.selectbox("Instrucción del cuidador",
                                           ['Sin estudios', 'Primaria', 'Secundaria',
                                            'Superior técnica', 'Superior universitaria'])
                motivo_ing = st.selectbox("Motivo de ingreso", ['Tratamiento', 'Control', 'Examen', 'Emergencia'])
            with g3:
                diagnostico = st.selectbox("Diagnóstico", [
                    'Leucemia linfoblástica aguda', 'Leucemia mieloide aguda',
                    'Linfoma de Hodgkin', 'Linfoma no Hodgkin', 'Tumor cerebral',
                    'Neuroblastoma', 'Tumor de Wilms', 'Osteosarcoma',
                    'Retinoblastoma', 'Rabdomiosarcoma'])
                etapa       = st.selectbox("Etapa del cáncer", ['I', 'II', 'III', 'IV'])
                tratamiento = st.selectbox("Tratamiento", ['Quimioterapia', 'Radioterapia', 'Cirugía',
                                                           'Quimio + Radio', 'Quimio + Cirugía'])

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            st.markdown('<p class="st">🩺 Situación clínica y de apoyo</p>', unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3)
            with d1:
                meses_trat   = st.number_input("Meses en tratamiento", 1, 36, 6)
                num_reing    = st.number_input("N° reingresos previos", 0, 7, 0)
                motivo_reing = st.selectbox("Motivo de reingreso", [
                    'Primer ingreso', 'Continuación de tratamiento', 'Recaída',
                    'Continuación de quimioterapia', 'Seguimiento médico',
                    'Continuación de radioterapia', 'Control médico'])
            with d2:
                estado_fis   = st.selectbox("Movilidad", ['Caminando', 'Con ayuda parcial',
                                                          'Permanece en cama', 'Usa silla de ruedas'])
                estado_nut   = st.selectbox("Estado nutricional", ['Normal', 'Desnutrición leve', 'Desnutrición severa'])
                infeccion    = st.selectbox("¿Infección activa?", ['No', 'Sí'])
                perdida_peso = st.selectbox("¿Pérdida de peso reciente?", ['No', 'Sí'])
            with d3:
                adherencia     = st.selectbox("¿Sigue el tratamiento?", ['Alta', 'Media', 'Baja'])
                apoyo          = st.selectbox("Apoyo familiar", ['Fuerte', 'Moderado', 'Limitado'])
                acceso_med     = st.selectbox("Acceso a medicamentos", ['Completo', 'Parcial', 'Limitado'])
                emocional      = st.selectbox("Estado emocional", ['Estable', 'Ansioso', 'Deprimido'])
                comorbilidades = st.number_input("Otras enfermedades (n°)", 0, 3, 0)
                frec_hosp      = st.number_input("Hospitalizaciones (últimos 3 meses)", 0, 8, 0)

            submitted = st.form_submit_button("Evaluar prioridad de atención", type="primary",
                                              width='stretch')

    with col_res:
        st.markdown('<p class="st">📊 Resultado</p>', unsafe_allow_html=True)

        if not submitted:
            st.markdown(f"""
            <div style="background:{BLANCO};border-radius:10px;padding:40px 20px;
                        text-align:center;color:#bbb;box-shadow:0 1px 6px rgba(0,0,0,0.08);">
              <div style="font-size:2.8rem;margin-bottom:10px">👤</div>
              <p style="font-size:0.93rem;color:#aaa">Completa los datos del paciente<br>
              y presiona <strong>Evaluar</strong>.</p>
            </div>""", unsafe_allow_html=True)

        else:
            ord_maps_p = {
                'etapa_cancer':               {'I': 0, 'II': 1, 'III': 2, 'IV': 3},
                'estado_fisico':              {'Caminando': 0, 'Con ayuda parcial': 1,
                                               'Permanece en cama': 2, 'Usa silla de ruedas': 3},
                'estado_nutricional':         {'Normal': 0, 'Desnutrición leve': 1, 'Desnutrición severa': 2},
                'adherencia_tratamiento':     {'Alta': 0, 'Media': 1, 'Baja': 2},
                'apoyo_familiar':             {'Fuerte': 0, 'Moderado': 1, 'Limitado': 2},
                'acceso_medicamentos':        {'Completo': 0, 'Parcial': 1, 'Limitado': 2},
                'estado_emocional_paciente':  {'Estable': 0, 'Ansioso': 1, 'Deprimido': 2},
                'grado_instruccion_cuidador': {'Sin estudios': 0, 'Primaria': 1, 'Secundaria': 2,
                                               'Superior técnica': 3, 'Superior universitaria': 4},
            }
            infecc_val = 1 if infeccion == 'Sí' else 0
            peso_val   = 1 if perdida_peso == 'Sí' else 0

            row = {
                'edad': edad, 'distancia_origen_km': distancia,
                'meses_en_tratamiento': meses_trat, 'num_reingresos': num_reing,
                'presencia_infeccion': infecc_val, 'frecuencia_hospitalizacion_3m': frec_hosp,
                'perdida_peso_reciente': peso_val, 'num_comorbilidades': comorbilidades,
                'etapa_cancer_enc':               ord_maps_p['etapa_cancer'][etapa],
                'estado_fisico_enc':              ord_maps_p['estado_fisico'][estado_fis],
                'estado_nutricional_enc':         ord_maps_p['estado_nutricional'][estado_nut],
                'adherencia_tratamiento_enc':     ord_maps_p['adherencia_tratamiento'][adherencia],
                'apoyo_familiar_enc':             ord_maps_p['apoyo_familiar'][apoyo],
                'acceso_medicamentos_enc':        ord_maps_p['acceso_medicamentos'][acceso_med],
                'estado_emocional_paciente_enc':  ord_maps_p['estado_emocional_paciente'][emocional],
                'grado_instruccion_cuidador_enc': ord_maps_p['grado_instruccion_cuidador'][instruccion],
            }

            all_features = m_pac['feat_cols']
            X_new = pd.DataFrame(0.0, index=[0], columns=all_features)
            for k, v in row.items():
                if k in X_new.columns: X_new.loc[0, k] = v
            for cat_col, val in [('sexo', sexo), ('diagnostico', diagnostico),
                                 ('tipo_tratamiento', tratamiento), ('motivo_ingreso', motivo_ing),
                                 ('motivo_reingreso', motivo_reing), ('lugar_procedencia', lugar)]:
                col_name = f"{cat_col}_{val}"
                if col_name in X_new.columns: X_new.loc[0, col_name] = 1.0

            X_sc   = m_pac['scaler'].transform(X_new.values)
            pred   = m_pac['modelo'].predict(X_sc)[0]
            probs  = m_pac['modelo'].predict_proba(X_sc)[0]
            clases = m_pac['clases']
            nivel  = clases[pred]

            cfg = {
                'Alto':  dict(css='r-alto',  color=ROJO,   icon='🔴', titulo='PRIORIDAD ALTA',
                              accion='Atender primero: avisar al médico tratante y hacer seguimiento diario.'),
                'Medio': dict(css='r-medio', color=DORADO, icon='🟡', titulo='PRIORIDAD MEDIA',
                              accion='Seguimiento semanal: revisar alimentación y que siga su tratamiento.'),
                'Bajo':  dict(css='r-bajo',  color=VERDE,  icon='🟢', titulo='PRIORIDAD BAJA',
                              accion='Continuar con los controles habituales según cronograma.'),
            }[nivel]

            st.markdown(f"""<div class="{cfg['css']}">
              <p class="r-tit" style="color:{cfg['color']}">{cfg['icon']} {cfg['titulo']}</p>
              <p class="r-acc"><strong>Qué hacer:</strong> {cfg['accion']}</p>
              <p class="r-acc" style="font-size:0.78rem;opacity:0.7;margin-top:6px">
                Resultado orientativo — la evaluación clínica la confirma el personal de salud.
              </p>
            </div>""", unsafe_allow_html=True)

            # Confianza de la evaluación
            orden    = ['Alto', 'Medio', 'Bajo']
            cols_bar = [ROJO, DORADO, VERDE]
            idx_ord  = [list(clases).index(c) for c in orden]
            vals     = [probs[i] * 100 for i in idx_ord]
            fig_b = go.Figure(go.Bar(
                x=['Alta', 'Media', 'Baja'], y=vals, marker_color=cols_bar,
                text=[f"{v:.0f}%" for v in vals], textposition='outside',
                textfont=dict(size=12, color=GRIS), cliponaxis=False,
            ))
            fig_b.update_layout(**CL(190, dict(l=4, r=4, t=22, b=4)),
                                xaxis=ax(False, ''), yaxis=ax(True, '%') | {'range': [0, 120]})
            st.plotly_chart(fig_b, width='stretch')

            # Factores de riesgo
            factores = []
            if etapa in ['III', 'IV']:        factores.append(f"Cáncer en etapa {etapa} (avanzada)")
            if estado_fis != 'Caminando':     factores.append(f"Movilidad reducida: {estado_fis.lower()}")
            if estado_nut != 'Normal':        factores.append(f"{estado_nut}")
            if infecc_val:                    factores.append("Infección activa")
            if peso_val:                      factores.append("Pérdida de peso reciente")
            if adherencia == 'Baja':          factores.append("No está siguiendo bien el tratamiento")
            if motivo_reing == 'Recaída':     factores.append("Reingresa por recaída")
            if apoyo == 'Limitado':           factores.append("Poco apoyo familiar")

            if factores:
                st.markdown(f"<p style='font-weight:600;color:{AZUL};margin:4px 0 6px'>⚠ Señales a vigilar:</p>",
                            unsafe_allow_html=True)
                for f in factores:
                    st.markdown(f"<div style='background:#f8f9fa;border-left:3px solid {DORADO};"
                                f"padding:5px 10px;border-radius:4px;margin-bottom:4px;"
                                f"font-size:0.85rem;color:{GRIS}'>• {f}</div>",
                                unsafe_allow_html=True)

    # ── Pacientes registrados con prioridad alta ─────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown('<p class="st">🔴 Pacientes registrados que requieren atención primero</p>',
                unsafe_allow_html=True)
    if len(pac_alto) == 0:
        st.success("Ningún paciente registrado con prioridad alta.")
    else:
        tabla_alto = pac_alto[['id_paciente', 'edad', 'diagnostico', 'etapa_cancer',
                               'estado_nutricional', 'presencia_infeccion',
                               'adherencia_tratamiento']].copy()
        tabla_alto['presencia_infeccion'] = tabla_alto['presencia_infeccion'].map({1: 'Sí', 0: 'No'})
        tabla_alto = tabla_alto.rename(columns={
            'id_paciente': 'Paciente', 'edad': 'Edad', 'diagnostico': 'Diagnóstico',
            'etapa_cancer': 'Etapa', 'estado_nutricional': 'Nutrición',
            'presencia_infeccion': 'Infección', 'adherencia_tratamiento': 'Sigue tratamiento'})
        st.dataframe(tabla_alto, width='stretch', hide_index=True, height=260)
        st.caption(f"{len(pac_alto)} pacientes con prioridad alta de {total_pac} registrados. "
                   "Lista ordenada por registro; confirmar cada caso con el equipo médico.")

    # Distribución general
    col_dist, col_ficha = st.columns([3, 2], gap="medium")
    with col_dist:
        st.markdown('<p class="st">👥 Todos los pacientes por nivel de prioridad</p>', unsafe_allow_html=True)
        vc2 = df_pac['nivel_riesgo'].value_counts().reindex(['Alto', 'Medio', 'Bajo'])
        for nk, ck, lbl in [('Alto', ROJO, 'Prioridad alta'), ('Medio', DORADO, 'Prioridad media'),
                            ('Bajo', VERDE, 'Prioridad baja')]:
            cnt = vc2[nk]; pct = cnt / total_pac * 100
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:7px">
              <div style="width:120px;font-weight:600;font-size:0.84rem;color:{ck}">{lbl}</div>
              <div style="flex:1;background:#e9ecef;border-radius:4px;height:14px;overflow:hidden">
                <div style="width:{pct:.0f}%;background:{ck};height:100%;border-radius:4px"></div>
              </div>
              <div style="width:72px;text-align:right;font-size:0.82rem;color:{GRIS}">{cnt} ({pct:.0f}%)</div>
            </div>""", unsafe_allow_html=True)
    with col_ficha:
        with st.expander("🔬 Ficha técnica del modelo (para el equipo técnico)"):
            if metricas_pac:
                df_cmp_p = pd.DataFrame(metricas_pac).T
                df_cmp_p.columns = ['F1-Macro', 'AUC-Macro', 'Accuracy']
                st.markdown(f"Modelo en producción: **{nombre_mod_pac}** (clasificación multiclase). "
                            "La prioridad se calcula con 24 variables clínicas y sociales; "
                            "el desbalanceo de clases se corrigió durante el entrenamiento.")
                st.dataframe(df_cmp_p.round(4), width='stretch')
