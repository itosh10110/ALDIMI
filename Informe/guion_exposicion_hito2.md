# 🎤 Guión de Exposición — Hito 2
## ALDIMI Predict | Machine Learning 1ACC0057 · UPC | Mayo 2026

---

## 🗂️ ESTRUCTURA DE LA EXPOSICIÓN (aprox. 15–20 min)

1. Contexto del problema
2. Qué hicimos en el Hito 2
3. Frente 1 — Inventario: EDA → Datos → Modelos → Resultados
4. Frente 2 — Pacientes: EDA → Datos → Modelos → Resultados
5. Dashboard interactivo (demo en vivo)
6. Conclusiones y próximos pasos

---

## 1. APERTURA — Contexto del problema

> *"ALDIMI — el Albergue Divina Misericordia — alberga a niños con cáncer que vienen de provincias a Lima para recibir tratamiento. Estos pacientes dependen completamente del albergue: alimentación, medicamentos, cama. Si falta un insumo, no hay dónde más ir. Si un paciente se deteriora sin que nadie lo note a tiempo, las consecuencias pueden ser irreversibles.*
>
> *Nuestro sistema, ALDIMI Predict, busca darle al albergue dos herramientas de anticipación: saber cuándo va a faltar stock, y saber qué pacientes están en mayor riesgo clínico."*

---

## 2. ¿QUÉ ES EL HITO 2?

En el Hito 2 completamos las **Fases 2, 3 y 4 de CRISP-DM** para los dos frentes del proyecto:

| Fase | Nombre | Qué hicimos |
|------|--------|-------------|
| Fase 2 | Comprensión de datos (EDA) | Analizamos distribuciones, calidad, correlaciones y balance de clases |
| Fase 3 | Preparación de datos | Encoding, escalado (MinMaxScaler), split train/test |
| Fase 4 | Modelado baseline | Entrenamos 3 algoritmos y los comparamos con métricas |

Y además entregamos un **dashboard interactivo en Streamlit** que permite usar los modelos en tiempo real.

---

## 3. FRENTE 1 — PREDICCIÓN DE INVENTARIO

### ¿Qué predice?
Si un artículo del almacén va a quedarse **sin stock en los próximos 7 o 14 días**.  
Es clasificación binaria: `0 = sin alerta` / `1 = alerta de quiebre`.

### 3.1 EDA — Lo que descubrimos en los datos

Trabajamos con **dos datasets**:
- `aldimi_dataset_completo.csv` → para exploración (artículos, categorías, tasa de rotación)
- `aldimi_dataset_semanal.csv` → para modelado (con variables temporales por semana)

**Hallazgos clave del EDA:**

- El almacén tiene artículos en múltiples categorías (alimentos, medicamentos, higiene, etc.). Graficamos la distribución por categoría con barras horizontales.
- Detectamos artículos con **existencias negativas** (stock agotado) y artículos con **tasa de rotación > 1** (salen más de lo que entran — señal de quiebre inminente).
- El balance de clases para la alerta a 7 días es moderado (no extremo), lo que facilita el modelado. Lo mismo para 14 días.
- El mapa de calor de correlaciones mostró que `tasa_rotacion`, `total_salidas` y `existencias_actuales` tienen alta correlación con la variable objetivo.

### 3.2 Preparación de datos

Las **features** que usamos para entrenar:
- `categoria` → codificada con LabelEncoder
- `ocupacion_albergue` (cuántas familias hay esa semana)
- `stock_inicio_semana`
- `ingresos_semana` (donaciones recibidas)
- `salidas_semana` (consumo)
- `rolling_avg_salidas_3sem` (promedio móvil de consumo últimas 3 semanas)
- `semana_del_año`

Todo escalado con **MinMaxScaler** (normalización a [0,1]).  
Split: **70% entrenamiento / 30% prueba**, estratificado por clase.

### 3.3 Modelos y resultados

Entrenamos tres algoritmos baseline:

| Modelo | F1 (7 días) | AUC (7 días) | F1 (14 días) | AUC (14 días) |
|--------|-------------|--------------|--------------|---------------|
| Naive Bayes | 0.842 | 0.796 | 0.863 | 0.772 |
| KNN (k=5) | 0.849 | 0.833 | 0.846 | 0.806 |
| **Árbol de Decisión (depth=5)** | **0.970** | **0.981** | **0.944** | **0.975** |

→ El **Árbol de Decisión** gana claramente en ambos horizontes.

**¿Por qué F1 y no solo Accuracy?**  
Porque el dataset tiene algo de desbalance. El F1 penaliza tanto falsos positivos (alerta innecesaria) como falsos negativos (no detectar un quiebre real). En un albergue, un falso negativo es crítico.

**¿Por qué AUC-ROC?**  
Mide la capacidad del modelo de distinguir entre las dos clases independientemente del umbral. Un AUC de 0.981 significa que el árbol separa casi perfectamente los casos con y sin quiebre.

---

## 4. FRENTE 2 — CLASIFICACIÓN DE RIESGO DE PACIENTES

### ¿Qué predice?
El **nivel de riesgo clínico** de un paciente: `Bajo / Medio / Alto`.  
Es clasificación multiclase con 3 categorías.

### 4.1 EDA — Lo que descubrimos

Dataset: `aldimi_pacientes_sintetico.csv`

**Hallazgos clave:**

- Balance de clases bastante equilibrado entre Bajo, Medio y Alto (≈33% cada uno).
- El gráfico de *nivel de riesgo por etapa del cáncer* mostró el patrón esperado: etapas III y IV concentran más pacientes de riesgo Alto.
- Variables como `etapa_cancer`, `estado_fisico`, `adherencia_tratamiento`, `apoyo_familiar`, `presencia_infeccion` y `perdida_peso_reciente` tienen correlación notable con el nivel de riesgo.
- Variables demográficas puras (edad, distancia) tienen correlación baja con el riesgo — el riesgo es clínico, no geográfico.

### 4.2 Preparación de datos

El dataset mezcla variables ordinales y nominales. Las tratamos así:

- **Encoding ordinal** para variables con jerarquía clara: `etapa_cancer` (I→0, II→1, III→2, IV→3), `estado_fisico`, `estado_nutricional`, `adherencia_tratamiento`, `apoyo_familiar`, `acceso_medicamentos`, `grado_instruccion_cuidador`.
- **One-Hot Encoding (OHE)** para variables nominales sin orden: `sexo`, `diagnostico`, `tipo_tratamiento`, `motivo_ingreso`, `lugar_procedencia`.
- Variables numéricas directas: `edad`, `distancia_origen_km`, `meses_en_tratamiento`, `num_reingresos`, `presencia_infeccion`, `frecuencia_hospitalizacion_3m`, `perdida_peso_reciente`, `num_comorbilidades`.
- **MinMaxScaler** + split 70/30 estratificado.

### 4.3 Modelos y resultados

| Modelo | Accuracy | F1-Macro | F1-Weighted | AUC-Macro |
|--------|----------|----------|-------------|-----------|
| Naive Bayes | (ver resultados del notebook) | — | — | — |
| KNN (k=5) | — | — | — | — |
| **Árbol de Decisión (depth=5)** | **mejor** | **mejor** | **mejor** | **mejor** |

→ El **Árbol de Decisión** vuelve a ser el mejor modelo baseline.

Usamos **F1-Macro** (promedio no ponderado entre clases) porque nos importa igualmente detectar bien el riesgo Bajo, Medio y Alto.  
Las **curvas ROC One-vs-Rest** muestran que el modelo distingue bien cada clase del resto.

---

## 5. DASHBOARD — DEMO EN VIVO

> *"Todo lo anterior lo integramos en una aplicación que cualquier coordinador del albergue puede usar, sin saber de machine learning."*

El dashboard tiene **3 secciones**:

### 🏠 Inicio
- KPIs en tiempo real: alertas de stock activas (7 y 14 días), pacientes en riesgo alto y bajo.
- Gráfico de artículos por categoría.
- Donut chart de distribución de riesgo de pacientes.
- Tabla de artículos críticos por tasa de rotación.
- Banner de alineación con ODS 2 (Hambre Cero) y ODS 3 (Salud y Bienestar).

### 📦 Alertas de Inventario
- Formulario donde se ingresa: categoría, stock actual, ingresos y consumo estimado, promedio de consumo, ocupación del albergue y semana del año.
- El modelo predice si habrá quiebre en 7 y 14 días, con probabilidad y recomendación de acción.
- Gauge charts de probabilidad por horizonte.
- Evolución semanal de stock por categoría (línea + barras de alertas).
- Top artículos por tasa de rotación.

### 👤 Evaluación de Pacientes
- Formulario completo con todos los campos clínicos del paciente.
- El modelo devuelve el nivel de riesgo (Bajo/Medio/Alto), la probabilidad de cada clase y los factores de riesgo detectados.
- Barra de distribución de la base de datos para contexto.

---

## 6. CONCLUSIONES Y PRÓXIMOS PASOS

### Conclusiones del Hito 2

- El **Árbol de Decisión (depth=5)** es el mejor modelo baseline en ambos frentes: F1=0.970 y AUC=0.981 para alertas de inventario a 7 días.
- Los datos de inventario tienen patrones claros y el modelo los captura bien. Los datos de pacientes requieren más cuidado con las variables categóricas, pero el modelo baseline ya da resultados sólidos.
- El dashboard demuestra que los modelos son utilizables en producción por personal no técnico.

### Para el Hito 3 haremos:
- **Random Forest** y **XGBoost** con tuning de hiperparámetros (GridSearch / RandomSearch).
- Validación cruzada estratificada.
- Interpretabilidad con SHAP (qué variables pesan más en cada predicción).
- Posiblemente: manejo de desbalance con SMOTE si los nuevos modelos lo requieren.

---

## 🗣️ POSIBLES PREGUNTAS Y CÓMO RESPONDERLAS

**¿Por qué usaron datos sintéticos para pacientes?**  
Por privacidad. Los datos reales de pacientes son sensibles. Generamos datos sintéticos con las mismas distribuciones que los formularios reales de ALDIMI para poder entrenar sin exponer información personal.

**¿Por qué solo 3 modelos?**  
Son los modelos baseline que pide el enunciado del hito. Son interpretables y nos permiten establecer un punto de partida. En el Hito 3 probamos modelos más potentes.

**¿Qué significa tasa de rotación?**  
Es cuántas unidades salen dividido entre el stock disponible. Si es > 1, significa que salen más unidades de las que hay — el artículo ya está agotado o en déficit.

**¿El dashboard funciona en tiempo real?**  
Sí, carga los datos de los CSV en tiempo real y usa los modelos entrenados (guardados como `.pkl`). Se puede actualizar con nuevos datos sin reentrenar.

**¿Cómo se alinea con los ODS?**  
ODS 2 (Hambre Cero): garantizar que siempre haya alimentos en el albergue. ODS 3 (Salud y Bienestar): detectar a tiempo a los pacientes que necesitan atención prioritaria.

---

*Guión preparado para la exposición del Hito 2 — ALDIMI Predict*
*ML 1ACC0057 · UPC · Mayo 2026*
