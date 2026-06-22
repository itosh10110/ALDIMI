import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib, warnings, os
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
PURPURA = "#7b1fa2"

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

MODEL_COLOR = {'Random Forest': CELESTE, 'XGBoost': PURPURA, 'Árbol (baseline)': '#90A4AE'}

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

.model-badge {{ display:inline-block; padding:3px 10px; border-radius:12px;
                font-size:0.78rem; font-weight:700; color:white; }}
.model-badge.rf  {{ background:{CELESTE}; }}
.model-badge.xgb {{ background:{PURPURA}; }}

.r-alto   {{ background:#fdf3f2; border:1.5px solid {ROJO};   border-radius:8px; padding:12px 16px; margin-bottom:10px; }}
.r-medio  {{ background:#fef9f0; border:1.5px solid {DORADO}; border-radius:8px; padding:12px 16px; margin-bottom:10px; }}
.r-bajo   {{ background:#f0faf4; border:1.5px solid {VERDE};  border-radius:8px; padding:12px 16px; margin-bottom:10px; }}
.r-alerta {{ background:#fdf3f2; border:1.5px solid {ROJO};   border-radius:8px; padding:12px 16px; margin-bottom:10px; }}
.r-ok     {{ background:#f0faf4; border:1.5px solid {VERDE};  border-radius:8px; padding:12px 16px; margin-bottom:10px; }}
.r-tit    {{ font-size:1.05rem; font-weight:700; margin:0 0 5px 0; }}
.r-acc    {{ font-size:0.86rem; margin:0; color:{GRIS}; line-height:1.5; }}

.hito-bar {{ display:flex; gap:4px; align-items:center; margin:10px 0; }}
.hito-step {{ flex:1; padding:5px 4px; text-align:center; border-radius:6px; font-size:0.72rem; }}
.hito-done {{ background:#d5f5e3; color:#1a7a4a; font-weight:700; }}
.hito-curr {{ background:#fff3cd; color:#856404; font-weight:700; border:1.5px solid {DORADO}; }}
.hito-pend {{ background:#e9ecef; color:#6c757d; }}

div.stButton>button, div.stFormSubmitButton>button {{
  background-color:{CELESTE}; color:white; border:none; border-radius:8px;
  padding:9px 22px; font-weight:600; font-size:0.92rem;
  width:100%; transition:background 0.18s;
}}
div.stButton>button:hover, div.stFormSubmitButton>button:hover {{ background-color:{AZUL}; color:white; }}

[data-baseweb="slider"] [role="slider"] {{ background-color:{CELESTE} !important; border-color:{CELESTE} !important; }}
div[data-baseweb="slider"] > div > div > div:nth-child(1) > div {{ background:{CELESTE} !important; }}
div[data-testid="stForm"] {{ border:none !important; padding:0 !important; background:transparent !important; box-shadow:none !important; }}
[data-testid="stDataFrame"] {{ border-radius:8px; overflow:hidden; }}
</style>
""", unsafe_allow_html=True)


# ── CARGA ─────────────────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    df_inv  = pd.read_csv(os.path.join(BASE, 'data', 'processed', 'aldimi_dataset_semanal.csv'))
    df_orig = pd.read_csv(os.path.join(BASE, 'data', 'processed', 'aldimi_dataset_completo.csv'))
    df_pac  = pd.read_csv(os.path.join(BASE, 'data', 'processed', 'aldimi_pacientes_sintetico.csv'))
    return df_inv, df_orig, df_pac

@st.cache_resource
def cargar_modelos():
    m_inv = joblib.load(os.path.join(BASE, 'models', 'models_inventario.pkl'))
    m_pac = joblib.load(os.path.join(BASE, 'models', 'models_pacientes.pkl'))
    return m_inv, m_pac

try:
    df_inv, df_orig, df_pac = cargar_datos()
except FileNotFoundError as e:
    st.error(f"❌ Archivos de datos no encontrados: {e}"); st.stop()
try:
    m_inv, m_pac = cargar_modelos()
except FileNotFoundError as e:
    st.error(f"❌ Modelos no encontrados: {e}"); st.stop()

total_pac       = len(df_pac)
alertas7_total  = int(df_inv['alerta_7_dias'].sum())
alto_riesgo_tot = int((df_pac['nivel_riesgo'] == 'Alto').sum())

# Nombres de modelo para mostrar
nombre_mod_inv = m_inv.get('mejor_nombre_7d', 'Árbol de Decisión')
nombre_mod_pac = m_pac.get('mejor_nombre', 'Árbol de Decisión')
badge_inv = 'rf' if 'Forest' in nombre_mod_inv else ('xgb' if 'XGB' in nombre_mod_inv else '')
badge_pac = 'rf' if 'Forest' in nombre_mod_pac else ('xgb' if 'XGB' in nombre_mod_pac else '')

# Métricas globales (disponibles en todas las páginas)
metricas_inv = m_inv.get('metricas_7d', {})
metricas_pac = m_pac.get('metricas', {})


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:8px 0 4px'>
      <span style='font-size:2rem'>🏥</span>
      <p style='font-weight:700;font-size:1.05rem;margin:4px 0 0'>ALDIMI Predict</p>
      <p style='font-size:0.78rem;opacity:0.75;margin:2px 0 0'>Sistema Predictivo de Gestión</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    pagina = st.radio("", [
        "🏠  Inicio",
        "📦  Alertas de Inventario",
        "👤  Evaluación de Pacientes",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"🔴 **{alertas7_total}** alertas de stock activas")
    st.markdown(f"🔴 **{alto_riesgo_tot}** pacientes en riesgo alto")
    st.markdown("---")
    # Fase del proyecto en sidebar
    st.markdown(f"""
    <div style='font-size:0.78rem;opacity:0.9;line-height:1.8'>
      <p style='font-weight:700;margin:0 0 4px'>📍 Fase del proyecto</p>
      <p style='margin:0'>✅ Hito 1 — EDA</p>
      <p style='margin:0'>✅ Hito 2 — Baseline</p>
      <p style='margin:0;color:#ffd700;font-weight:700'>🔄 Hito 3 — Avanzado ←</p>
      <p style='margin:0;opacity:0.5'>⬜ Hito 4 — Final</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"""
    <p style='font-size:0.76rem;opacity:0.65'>
      Modelos activos:<br>
      📦 {nombre_mod_inv}<br>
      👤 {nombre_mod_pac}<br><br>
      ML 1ACC0057 · UPC<br>Junio 2026
    </p>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# INICIO
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "🏠  Inicio":

    st.markdown("""<div class='ph'>
      <h1>🏥 ALDIMI Predict</h1>
      <p>Sistema Predictivo de Gestión de Recursos y Riesgos · Albergue Divina Misericordia</p>
    </div>""", unsafe_allow_html=True)

    # ── Banner de fase del proyecto ───────────────────────────────────────────
    f1_base_inv  = metricas_inv.get('Árbol (baseline)', {}).get('f1', 0.970)
    f1_mejor_inv = metricas_inv.get(nombre_mod_inv, {}).get('f1', 0)
    f1_base_pac  = metricas_pac.get('Árbol (baseline)', {}).get('f1_macro', 0.606)
    f1_mejor_pac = metricas_pac.get(nombre_mod_pac, {}).get('f1_macro', 0)
    delta_inv = f1_mejor_inv - f1_base_inv
    delta_pac = f1_mejor_pac - f1_base_pac

    st.markdown(f"""
    <div style="background:linear-gradient(90deg,#fff8e1,#fffde7);border:1.5px solid {DORADO};
                border-radius:10px;padding:12px 18px;margin-bottom:14px;">
      <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
        <div style="font-weight:700;font-size:0.95rem;color:#856404;">
          🔄 Hito 3 en curso — Modelado Avanzado &amp; Interpretabilidad (Semana 12)
        </div>
        <div class="hito-bar" style="flex:1;min-width:260px;">
          <div class="hito-step hito-done">✓ H1<br>EDA</div>
          <div class="hito-step hito-done">✓ H2<br>Baseline</div>
          <div class="hito-step hito-curr">▶ H3<br>Avanzado</div>
          <div class="hito-step hito-pend">H4<br>Final</div>
        </div>
      </div>
      <div style="margin-top:10px;display:flex;gap:24px;flex-wrap:wrap;font-size:0.83rem;">
        <span>📦 <b>Inventario</b>: {nombre_mod_inv}
          — F1 = <b style="color:{VERDE}">{f1_mejor_inv:.3f}</b>
          <span style="color:{VERDE}"> (+{delta_inv:.3f} vs baseline)</span>
        </span>
        <span>👤 <b>Pacientes</b>: {nombre_mod_pac}
          — F1-Macro = <b style="color:{VERDE}">{f1_mejor_pac:.3f}</b>
          <span style="color:{VERDE}"> (+{delta_pac:.3f} vs baseline)</span>
        </span>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── KPIs ─────────────────────────────────────────────────────────────────
    alertas7  = int(df_inv['alerta_7_dias'].sum())
    alertas14 = int(df_inv['alerta_14_dias'].sum())
    n_alto    = int((df_pac['nivel_riesgo'] == 'Alto').sum())
    n_bajo    = int((df_pac['nivel_riesgo'] == 'Bajo').sum())

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="kpi danger">
          <p class="kpi-num">{alertas7:,}</p>
          <p class="kpi-label">⚠ Alertas stock · 7 días</p>
          <p class="kpi-sub">{alertas7/len(df_inv)*100:.0f}% de los registros</p>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi warn">
          <p class="kpi-num">{alertas14:,}</p>
          <p class="kpi-label">⚠ Alertas stock · 14 días</p>
          <p class="kpi-sub">{alertas14/len(df_inv)*100:.0f}% de los registros</p>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi danger">
          <p class="kpi-num">{n_alto}</p>
          <p class="kpi-label">🔴 Pacientes riesgo alto</p>
          <p class="kpi-sub">De {total_pac} registrados</p>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi ok">
          <p class="kpi-num">{n_bajo}</p>
          <p class="kpi-label">🟢 Pacientes riesgo bajo</p>
          <p class="kpi-sub">{n_bajo/total_pac*100:.0f}% del total</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Gráficos + comparativa de modelos ────────────────────────────────────
    col_a, col_b = st.columns(2, gap="medium")

    with col_a:
        st.markdown('<p class="st">📦 Artículos por categoría</p>', unsafe_allow_html=True)
        cats = df_orig['categoria'].value_counts().head(10)
        fig_cats = go.Figure(go.Bar(
            x=cats.values, y=cats.index, orientation='h',
            marker_color=CELESTE, text=cats.values, textposition='auto',
            insidetextanchor='end', textfont=dict(size=12, color=BLANCO), cliponaxis=False,
        ))
        fig_cats.update_layout(**CL(300, dict(l=4, r=16, t=8, b=4)),
                               xaxis=ax(True, 'N° artículos'),
                               yaxis=ax(False, '') | dict(automargin=True))
        st.plotly_chart(fig_cats, use_container_width=True)

    with col_b:
        st.markdown('<p class="st">👥 Distribución de riesgo de pacientes</p>', unsafe_allow_html=True)
        vc = df_pac['nivel_riesgo'].value_counts().reindex(['Bajo','Medio','Alto'])
        fig_donut = go.Figure(go.Pie(
            labels=vc.index, values=vc.values, hole=0.55,
            marker=dict(colors=[VERDE, NARANJA, ROJO], line=dict(color=FONDO, width=3)),
            textinfo='label+percent', textfont=dict(size=12, color=GRIS), pull=[0, 0, 0.05],
        ))
        fig_donut.update_layout(
            **CL(300, dict(l=4, r=4, t=8, b=30), show_legend=True),
            legend=dict(orientation='h', y=-0.06, x=0.5, xanchor='center', font=dict(size=11, color=GRIS)),
            annotations=[dict(text=f'<b>{total_pac}</b><br><span style="font-size:10px">pacientes</span>',
                              x=0.5, y=0.5, showarrow=False, font_size=15)],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── Comparativa baseline vs avanzados ─────────────────────────────────────
    st.markdown('<p class="st">📊 Comparativa de modelos — Hito 2 (baseline) vs Hito 3 (avanzados)</p>',
                unsafe_allow_html=True)
    col_cmp1, col_cmp2 = st.columns(2, gap="medium")

    with col_cmp1:
        nombres_i = list(metricas_inv.keys())
        f1s_i     = [metricas_inv[n]['f1']  for n in nombres_i]
        aucs_i    = [metricas_inv[n]['auc'] for n in nombres_i]
        colors_i  = [MODEL_COLOR.get(n, '#90A4AE') for n in nombres_i]
        fig_ci = go.Figure()
        fig_ci.add_trace(go.Bar(name='F1-Score', x=nombres_i, y=f1s_i,
            marker_color=colors_i, text=[f'{v:.3f}' for v in f1s_i],
            textposition='outside', cliponaxis=False,
            textfont=dict(size=10, color=GRIS)))
        fig_ci.update_layout(
            **CL(240, dict(l=4, r=4, t=30, b=4)),
            title=dict(text='📦 Inventario 7d — F1 Test', font=dict(size=12, color=AZUL)),
            xaxis=ax(False, '') | {'tickfont': dict(size=9, color=GRIS)},
            yaxis=ax(True, 'F1') | {'range': [0.88, 1.02]},
        )
        st.plotly_chart(fig_ci, use_container_width=True)

    with col_cmp2:
        nombres_p = list(metricas_pac.keys())
        f1s_p     = [metricas_pac[n]['f1_macro'] for n in nombres_p]
        colors_p  = [MODEL_COLOR.get(n, '#90A4AE') for n in nombres_p]
        fig_cp = go.Figure()
        fig_cp.add_trace(go.Bar(name='F1-Macro', x=nombres_p, y=f1s_p,
            marker_color=colors_p, text=[f'{v:.3f}' for v in f1s_p],
            textposition='outside', cliponaxis=False,
            textfont=dict(size=10, color=GRIS)))
        fig_cp.update_layout(
            **CL(240, dict(l=4, r=4, t=30, b=4)),
            title=dict(text='👤 Pacientes — F1-Macro Test', font=dict(size=12, color=AZUL)),
            xaxis=ax(False, '') | {'tickfont': dict(size=9, color=GRIS)},
            yaxis=ax(True, 'F1-Macro') | {'range': [0.4, 0.9]},
        )
        st.plotly_chart(fig_cp, use_container_width=True)

    # ── Tabla artículos críticos ───────────────────────────────────────────────
    st.markdown('<p class="st">🚨 Artículos con mayor riesgo de quiebre de stock</p>', unsafe_allow_html=True)
    top_c = df_orig.nlargest(8, 'tasa_rotacion')[
        ['codigo_articulo','categoria','existencias_actuales','tasa_rotacion']].copy()
    top_c['Estado'] = top_c['tasa_rotacion'].apply(
        lambda x: '🔴 Agotado' if x > 1 else ('🟠 Crítico' if x > 0.5 else '🟡 Monitoreo'))
    top_c['tasa_rotacion']       = top_c['tasa_rotacion'].round(2)
    top_c['existencias_actuales'] = top_c['existencias_actuales'].round(1)
    top_c = top_c.rename(columns={'codigo_articulo':'Código','categoria':'Categoría',
                                   'existencias_actuales':'Stock actual','tasa_rotacion':'Tasa rotación'})
    st.dataframe(top_c, use_container_width=True, hide_index=True)

    st.markdown(f"""<div style="background:{AZUL};color:white;border-radius:8px;padding:10px 18px;
                font-size:0.83rem;margin-top:10px;line-height:1.75">
      <strong>🎯 Alineación ODS</strong> &nbsp;·&nbsp;
      <strong>ODS 2 — Hambre Cero:</strong> Disponibilidad continua de alimentos para pacientes oncológicos. &nbsp;·&nbsp;
      <strong>ODS 3 — Salud y Bienestar:</strong> Identificación temprana de pacientes en riesgo clínico.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ALERTAS DE INVENTARIO
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📦  Alertas de Inventario":

    badge_cls = 'rf' if 'Forest' in nombre_mod_inv else 'xgb'
    st.markdown(f"""<div class='ph'>
      <h1>📦 Alertas de Inventario
        <span class="model-badge {badge_cls}" style="vertical-align:middle;margin-left:10px;font-size:0.75rem;">
          {nombre_mod_inv}
        </span>
      </h1>
      <p>Predice quiebre de stock a 7 o 14 días · Modelo activo: <b>{nombre_mod_inv}</b>
         (F1={metricas_inv.get(nombre_mod_inv,{}).get('f1',0):.3f} · AUC={metricas_inv.get(nombre_mod_inv,{}).get('auc',0):.3f})</p>
    </div>""", unsafe_allow_html=True)

    # ── Predictor ─────────────────────────────────────────────────────────────
    with st.container():
        st.markdown('<p class="st">🔍 Consultar artículo del almacén</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([2, 2, 1], gap="medium")
        with c1:
            cat_sel      = st.selectbox("Categoría del artículo", m_inv['categorias'])
            stock_inicio = st.number_input("Stock disponible (unidades)", -50.0, 500.0, 8.0, 1.0)
            ingresos     = st.number_input("Ingresos estimados esta semana", 0.0, 200.0, 0.0, 1.0)
        with c2:
            salidas     = st.number_input("Consumo estimado esta semana", 0.0, 100.0, 3.0, 0.5)
            rolling_avg = st.number_input("Promedio consumo últimas 3 semanas", 0.0, 50.0, 2.5, 0.5)
            ocupacion   = st.slider("Familias en el albergue", 40, 100, 55)
        with c3:
            semana    = st.slider("Semana del año", 1, 52, 26)
            st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
            consultar = st.button("🔍 Evaluar quiebre", type="primary")

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
            lbl = 'QUIEBRE PROBABLE' if pred else 'Sin riesgo'
            ico = '⚠️' if pred else '✅'
            acc = ('Solicitar reposición urgente.' if dias == 7
                   else 'Planificar donaciones próxima semana.') if pred else \
                  ('Stock suficiente esta semana.' if dias == 7
                   else 'Stock proyectado suficiente a 14 días.')
            st.markdown(f"""<div class="{css}">
              <p class="r-tit" style="color:{col}">{ico} {dias} días — {lbl}</p>
              <p class="r-acc">Probabilidad: <strong>{prob*100:.1f}%</strong></p>
              <p class="r-acc" style="margin-top:5px">→ {acc}</p>
            </div>""", unsafe_allow_html=True)

        with r1: tarjeta_inv(pred_7,  prob_7,  7)
        with r2: tarjeta_inv(pred_14, prob_14, 14)
        with r3:
            fig_g = make_subplots(rows=1, cols=2,
                                  specs=[[{'type':'indicator'},{'type':'indicator'}]])
            for ci, (label, prob, pred) in enumerate(
                    [('7 días', prob_7, pred_7), ('14 días', prob_14, pred_14)], 1):
                bar_c = ROJO if pred else VERDE
                fig_g.add_trace(go.Indicator(
                    mode="gauge+number", value=round(prob * 100, 1),
                    number={'suffix': '%', 'font': {'size': 22, 'color': AZUL}},
                    title={'text': f"<b>{label}</b>", 'font': {'size': 11, 'color': GRIS}},
                    gauge={'axis': {'range': [0,100], 'tickfont': {'size': 9}},
                           'bar': {'color': bar_c, 'thickness': 0.28},
                           'bgcolor': 'rgba(248,249,250,1)',
                           'steps': [{'range': [0,40],'color':'#d5f5e3'},
                                     {'range': [40,70],'color':'#fdebd0'},
                                     {'range': [70,100],'color':'#fadbd8'}],
                           'threshold': {'line': {'color': AZUL, 'width': 2}, 'value': 50}}
                ), row=1, col=ci)
            fig_g.update_layout(height=190, paper_bgcolor='rgba(0,0,0,0)',
                                margin=dict(l=10, r=10, t=28, b=8),
                                font=dict(family='Segoe UI', color=GRIS))
            st.plotly_chart(fig_g, use_container_width=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Comparativa modelos (inventario) ──────────────────────────────────────
    with st.expander("📊 Ver comparativa de modelos — Inventario 7d", expanded=False):
        if metricas_inv:
            df_cmp = pd.DataFrame(metricas_inv).T.reset_index()
            df_cmp.columns = ['Modelo','F1-Score','AUC-ROC']
            df_cmp['Mejor'] = df_cmp['Modelo'] == nombre_mod_inv
            nombres_c = df_cmp['Modelo'].tolist()
            f1s_c     = df_cmp['F1-Score'].tolist()
            aucs_c    = df_cmp['AUC-ROC'].tolist()
            colors_c  = [MODEL_COLOR.get(n,'#90A4AE') for n in nombres_c]
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(name='F1-Score', x=nombres_c, y=f1s_c,
                marker_color=colors_c, text=[f'{v:.4f}' for v in f1s_c],
                textposition='outside', cliponaxis=False, textfont=dict(size=10, color=GRIS)))
            fig_cmp.add_trace(go.Scatter(name='AUC-ROC', x=nombres_c, y=aucs_c,
                mode='markers+lines', marker=dict(size=10, color=AZUL), line=dict(color=AZUL, dash='dot')))
            fig_cmp.update_layout(**CL(250, dict(l=4,r=4,t=30,b=4), show_legend=True),
                xaxis=ax(False,''), yaxis=ax(True,'') | {'range':[0.85,1.02]},
                legend=dict(orientation='h', y=1.12, x=0, font=dict(size=10, color=GRIS)))
            st.plotly_chart(fig_cmp, use_container_width=True)
            st.dataframe(df_cmp[['Modelo','F1-Score','AUC-ROC']].set_index('Modelo'),
                         use_container_width=True)

    # ── Evolución + Top artículos ──────────────────────────────────────────────
    col_ev, col_top = st.columns([3, 2], gap="medium")

    with col_ev:
        st.markdown('<p class="st">📈 Evolución de stock semanal</p>', unsafe_allow_html=True)
        cat_sel2 = st.selectbox("Categoría", sorted(df_inv['categoria'].unique()), key='cat2')
        df_cat = (df_inv[df_inv['categoria'] == cat_sel2]
                  .groupby('semana_del_año')
                  .agg(stock_promedio=('stock_fin_semana','mean'),
                       alertas_activas=('alerta_7_dias','sum'))
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
            name='Alertas 7d', marker_color='rgba(192,57,43,0.40)',
        ), secondary_y=True)
        fig_ev.add_hline(y=0, line_dash='dash', line_color=ROJO, line_width=1.2,
                         annotation_text='Stock 0', annotation_font_color=ROJO, annotation_font_size=9)
        fig_ev.update_layout(**CL(310, dict(l=4,r=4,t=32,b=4), show_legend=True),
                             xaxis=ax(True, 'Semana del año'),
                             legend=dict(orientation='h',y=1.1,x=0,font=dict(size=10,color=GRIS)))
        fig_ev.update_yaxes(title_text="Stock (uds)", secondary_y=False,
                             showgrid=True, gridcolor=GRID, tickfont=dict(color=GRIS), title_font=dict(color=GRIS))
        fig_ev.update_yaxes(title_text="Alertas", secondary_y=True,
                             showgrid=False, tickfont=dict(color=GRIS), title_font=dict(color=GRIS))
        st.plotly_chart(fig_ev, use_container_width=True)

    with col_top:
        st.markdown('<p class="st">🚨 Top artículos por tasa de rotación</p>', unsafe_allow_html=True)
        top15 = df_orig.nlargest(12, 'tasa_rotacion')[['codigo_articulo','tasa_rotacion']].copy()
        cap = float(np.percentile(top15['tasa_rotacion'], 90)) * 1.15
        top15['valor_vis'] = top15['tasa_rotacion'].clip(upper=cap)
        top15['label']     = top15['tasa_rotacion'].apply(lambda v: f"{v:.2f} ⚡" if v > cap else f"{v:.2f}")
        colors = [ROJO if v > 1 else (DORADO if v > 0.5 else CELESTE) for v in top15['tasa_rotacion']]
        fig_top = go.Figure(go.Bar(
            x=top15['valor_vis'], y=top15['codigo_articulo'], orientation='h',
            marker_color=colors, text=top15['label'], textposition='outside',
            textfont=dict(size=10, color=GRIS), cliponaxis=False,
        ))
        fig_top.add_vline(x=min(1.0, cap), line_dash='dash', line_color=ROJO, line_width=1.3,
                          annotation_text='Crítico', annotation_font_color=ROJO,
                          annotation_font_size=9, annotation_position='top right')
        fig_top.update_layout(**CL(310, dict(l=4,r=52,t=10,b=4), show_legend=False),
                              xaxis=ax(True, 'Tasa rotación') | {'range': [0, cap * 1.12]},
                              yaxis=ax(False,'') | {'automargin':True,'tickfont':dict(size=9,color=GRIS)})
        st.plotly_chart(fig_top, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUACIÓN DE PACIENTES
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "👤  Evaluación de Pacientes":

    badge_cls_p = 'rf' if 'Forest' in nombre_mod_pac else 'xgb'
    f1m_pac = metricas_pac.get(nombre_mod_pac, {}).get('f1_macro', 0)
    auc_pac = metricas_pac.get(nombre_mod_pac, {}).get('auc_macro', 0)
    st.markdown(f"""<div class='ph'>
      <h1>👤 Evaluación de Riesgo de Pacientes
        <span class="model-badge {badge_cls_p}" style="vertical-align:middle;margin-left:10px;font-size:0.75rem;">
          {nombre_mod_pac}
        </span>
      </h1>
      <p>Clasifica el nivel de riesgo clínico: Bajo · Medio · Alto · Modelo activo: <b>{nombre_mod_pac}</b>
         (F1-Macro={f1m_pac:.3f} · AUC={auc_pac:.3f})</p>
    </div>""", unsafe_allow_html=True)

    col_form, col_res = st.columns([3, 2], gap="medium")

    with col_form:
        with st.form("form_paciente", clear_on_submit=False):
            st.markdown('<p class="st">📋 Datos generales</p>', unsafe_allow_html=True)
            g1, g2, g3 = st.columns(3)
            with g1:
                edad       = st.number_input("Edad (años)", 2, 17, 8)
                sexo       = st.selectbox("Sexo", ["Masculino", "Femenino"])
                distancia  = st.number_input("Distancia a Lima (km)", 10, 1400, 600)
            with g2:
                lugar      = st.selectbox("Procedencia", ['Sierra sur','Sierra norte','Sierra centro',
                                                           'Selva','Costa norte','Costa sur','Lima'])
                instruccion = st.selectbox("Instrucción cuidador",
                                           ['Sin estudios','Primaria','Secundaria',
                                            'Superior técnica','Superior universitaria'])
                motivo_ing = st.selectbox("Motivo ingreso", ['Tratamiento','Control','Examen','Emergencia'])
            with g3:
                diagnostico = st.selectbox("Diagnóstico", [
                    'Leucemia linfoblástica aguda','Leucemia mieloide aguda',
                    'Linfoma de Hodgkin','Linfoma no Hodgkin','Tumor cerebral',
                    'Neuroblastoma','Tumor de Wilms','Osteosarcoma',
                    'Retinoblastoma','Rabdomiosarcoma'])
                etapa       = st.selectbox("Etapa del cáncer", ['I','II','III','IV'])
                tratamiento = st.selectbox("Tratamiento", ['Quimioterapia','Radioterapia','Cirugía',
                                                           'Quimio + Radio','Quimio + Cirugía'])

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            st.markdown('<p class="st">🩺 Datos clínicos</p>', unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3)
            with d1:
                meses_trat   = st.number_input("Meses en tratamiento", 1, 36, 6)
                num_reing    = st.number_input("N° reingresos previos", 0, 7, 0)
                motivo_reing = st.selectbox("Motivo reingreso", [
                    'Primer ingreso','Continuación de tratamiento','Recaída',
                    'Continuación de quimioterapia','Seguimiento médico',
                    'Continuación de radioterapia','Control médico'])
            with d2:
                estado_fis   = st.selectbox("Estado físico", ['Caminando','Con ayuda parcial',
                                                               'Permanece en cama','Usa silla de ruedas'])
                estado_nut   = st.selectbox("Estado nutricional", ['Normal','Desnutrición leve','Desnutrición severa'])
                infeccion    = st.selectbox("Infección activa", ['No','Sí'])
                perdida_peso = st.selectbox("Pérdida de peso", ['No','Sí'])
            with d3:
                adherencia     = st.selectbox("Adherencia tratamiento", ['Alta','Media','Baja'])
                apoyo          = st.selectbox("Apoyo familiar", ['Fuerte','Moderado','Limitado'])
                acceso_med     = st.selectbox("Acceso medicamentos", ['Completo','Parcial','Limitado'])
                emocional      = st.selectbox("Estado emocional", ['Estable','Ansioso','Deprimido'])
                comorbilidades = st.number_input("N° comorbilidades", 0, 3, 0)
                frec_hosp      = st.number_input("Hospitalizaciones (3 meses)", 0, 8, 0)

            submitted = st.form_submit_button("🔍 Evaluar nivel de riesgo", type="primary",
                                              use_container_width=True)

    with col_res:
        st.markdown('<p class="st">📊 Resultado de la evaluación</p>', unsafe_allow_html=True)

        if not submitted:
            st.markdown(f"""
            <div style="background:{BLANCO};border-radius:10px;padding:40px 20px;
                        text-align:center;color:#bbb;box-shadow:0 1px 6px rgba(0,0,0,0.08);">
              <div style="font-size:2.8rem;margin-bottom:10px">👤</div>
              <p style="font-size:0.93rem;color:#aaa">Completa los datos del paciente<br>
              y presiona <strong>Evaluar</strong> para obtener el resultado.</p>
            </div>""", unsafe_allow_html=True)

        else:
            ord_maps_p = {
                'etapa_cancer':               {'I':0,'II':1,'III':2,'IV':3},
                'estado_fisico':              {'Caminando':0,'Con ayuda parcial':1,
                                              'Permanece en cama':2,'Usa silla de ruedas':3},
                'estado_nutricional':         {'Normal':0,'Desnutrición leve':1,'Desnutrición severa':2},
                'adherencia_tratamiento':     {'Alta':0,'Media':1,'Baja':2},
                'apoyo_familiar':             {'Fuerte':0,'Moderado':1,'Limitado':2},
                'acceso_medicamentos':        {'Completo':0,'Parcial':1,'Limitado':2},
                'estado_emocional_paciente':  {'Estable':0,'Ansioso':1,'Deprimido':2},
                'grado_instruccion_cuidador': {'Sin estudios':0,'Primaria':1,'Secundaria':2,
                                              'Superior técnica':3,'Superior universitaria':4},
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
                'estado_emoc_enc':                ord_maps_p['estado_emocional_paciente'][emocional],
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
                'Alto':  dict(css='r-alto',  color=ROJO,   icon='🔴',
                              accion='Monitoreo intensivo. Notificar al médico tratante. Evaluación clínica prioritaria.'),
                'Medio': dict(css='r-medio', color=DORADO, icon='🟡',
                              accion='Seguimiento semanal. Verificar adherencia y estado nutricional en próxima visita.'),
                'Bajo':  dict(css='r-bajo',  color=VERDE,  icon='🟢',
                              accion='Continuar protocolo estándar. Próximo control según cronograma habitual.'),
            }[nivel]

            st.markdown(f"""<div class="{cfg['css']}">
              <p class="r-tit" style="color:{cfg['color']}">{cfg['icon']} Nivel de Riesgo: <strong>{nivel.upper()}</strong></p>
              <p class="r-acc"><strong>Acción recomendada:</strong> {cfg['accion']}</p>
              <p class="r-acc" style="font-size:0.78rem;opacity:0.7;margin-top:6px">
                Modelo: {nombre_mod_pac}
              </p>
            </div>""", unsafe_allow_html=True)

            # Gráfico de probabilidades
            orden    = ['Alto', 'Medio', 'Bajo']
            cols_bar = [ROJO, DORADO, VERDE]
            idx_ord  = [list(clases).index(c) for c in orden]
            vals     = [probs[i] * 100 for i in idx_ord]
            fig_b = go.Figure(go.Bar(
                x=orden, y=vals, marker_color=cols_bar,
                text=[f"{v:.1f}%" for v in vals], textposition='outside',
                textfont=dict(size=12, color=GRIS), cliponaxis=False,
            ))
            fig_b.update_layout(**CL(190, dict(l=4,r=4,t=22,b=4)),
                                xaxis=ax(False,''), yaxis=ax(True,'%') | {'range':[0,120]})
            st.plotly_chart(fig_b, use_container_width=True)

            # Factores de riesgo
            factores = []
            if etapa in ['III','IV']:       factores.append(f"Etapa {etapa} (avanzada)")
            if estado_fis != 'Caminando':   factores.append(f"Estado físico: {estado_fis}")
            if estado_nut != 'Normal':      factores.append(f"Nutrición: {estado_nut}")
            if infecc_val:                  factores.append("Infección activa")
            if peso_val:                    factores.append("Pérdida de peso reciente")
            if adherencia == 'Baja':        factores.append("Baja adherencia al tratamiento")
            if motivo_reing == 'Recaída':   factores.append("Ingreso por recaída")
            if apoyo == 'Limitado':         factores.append("Apoyo familiar limitado")

            if factores:
                st.markdown(f"<p style='font-weight:600;color:{AZUL};margin:4px 0 6px'>⚠ Factores detectados:</p>",
                            unsafe_allow_html=True)
                for f in factores:
                    st.markdown(f"<div style='background:#f8f9fa;border-left:3px solid {DORADO};"
                                f"padding:5px 10px;border-radius:4px;margin-bottom:4px;"
                                f"font-size:0.85rem;color:{GRIS}'>• {f}</div>",
                                unsafe_allow_html=True)

        # Comparativa modelos pacientes
        with st.expander("📊 Ver comparativa de modelos — Pacientes", expanded=False):
            if metricas_pac:
                df_cmp_p = pd.DataFrame(metricas_pac).T.reset_index()
                df_cmp_p.columns = ['Modelo','F1-Macro','AUC-Macro','Accuracy']
                st.dataframe(df_cmp_p.set_index('Modelo').round(4), use_container_width=True)

        # Distribución de pacientes registrados
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<p class="st">👥 Distribución en base de datos</p>', unsafe_allow_html=True)
        vc2 = df_pac['nivel_riesgo'].value_counts().reindex(['Alto','Medio','Bajo'])
        for nk, ck in [('Alto', ROJO), ('Medio', DORADO), ('Bajo', VERDE)]:
            cnt = vc2[nk]; pct = cnt / total_pac * 100
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:7px">
              <div style="width:56px;font-weight:600;font-size:0.84rem;color:{ck}">{nk}</div>
              <div style="flex:1;background:#e9ecef;border-radius:4px;height:14px;overflow:hidden">
                <div style="width:{pct:.0f}%;background:{ck};height:100%;border-radius:4px"></div>
              </div>
              <div style="width:72px;text-align:right;font-size:0.82rem;color:{GRIS}">{cnt} ({pct:.0f}%)</div>
            </div>""", unsafe_allow_html=True)
