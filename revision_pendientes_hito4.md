# Revisión de pendientes — ALDIMI Predict (Hito 4)

**Preparado para:** apoyar la revisión interna del equipo antes de la sustentación
**Fecha del análisis:** 6 de julio de 2026
**Fuentes cruzadas:** enunciado oficial (`TP-TF-1ACC0057-Enunciado`), chat del equipo (2–5 jul), y el estado real del código en el repo (rama `feat/ui-dashboard`, working directory actual)

Leyenda: ✅ Resuelto&nbsp;&nbsp;·&nbsp;&nbsp;⚠️ Parcial / verificar con el equipo&nbsp;&nbsp;·&nbsp;&nbsp;❌ Pendiente&nbsp;&nbsp;·&nbsp;&nbsp;🔧 Prototipo listo, falta revisión del equipo

> **Actualización (6 jul, tarde):** se corrigió el typo `MedioMedio` en el CSV de
> pacientes, se sincronizaron `guion_video_y_exposicion_hito4.md` y
> `analisis_arquitectura_ALDIMI.md` con el estado actual (modelo binario,
> métricas reales validadas por CV), se marcó `dashboard_legacy.py` como legacy,
> y se construyó un **prototipo del modelo de cantidad de inventario** que
> pedía Leonel — ver sección 2 y `notebooks/modelo_cantidad_inventario.ipynb`.
>
> **Actualización (6 jul, noche):** Jairo dio feedback nuevo y puntual (vía un
> compañero): (1) el consumo debe recalcularse automáticamente según el número
> de familias en el albergue, (2) cero gráficos/scores/tecnicismos en la vista
> del personal, todo en lenguaje operativo directo, y (3) el dashboard debe
> permitir tanto ver como **introducir** datos (inventario y pacientes). Se
> construyó `dashboard_operativo.py` — un **prototipo aparte** (no se tocó el
> `dashboard.py` del equipo, para no chocar con su trabajo en curso) que cubre
> los 3 puntos con lógica ya probada contra los datos reales. Ver sección 2bis.
> Sigue pendiente el punto más urgente: **commitear y pushear todo**.

---

## 0. Lo primero: riesgo de proceso, no de contenido

El repo está en la rama `feat/ui-dashboard`, pero esa rama **no tiene ningún commit propio**: apunta al mismo commit que `main` y `origin/main` (`91c0b2e "nuevo modelo para pacientes"`, 5 jul). Todo el trabajo del "full UI/UX" que se pidió el 4 jul —el nuevo `dashboard.py`, los datasets reprocesados, los scripts, el informe— existe **solo como cambios sin commitear en el working directory** (39 archivos modificados, prácticamente reescritos por completo).

Esto significa que si esta máquina se pierde, se resetea, o alguien más del equipo clona el repo desde GitHub, **nadie ve nada de este avance**. Es el punto más urgente de todos, antes que cualquier tema de UI o modelos:

1. `git add -A && git commit -m "..."` 
2. `git push origin feat/ui-dashboard` (o mergear a `main` si ya está estable)

---

## 1. Feedback de Jairo (2 jul) — estado real

| # | Jairo pidió | Estado | Evidencia |
|---|---|---|---|
| 1 | Unidad de medida estándar por producto | ⚠️ **Parcial** | `catalogo_productos.csv` ya tiene columna `unidad_medida` (kg/litro/lata/caja/paquete/unidad) y el dashboard la muestra. **Pero** los valores de consumo semanal (`salidas_semana`) siguen sin tener sentido para unidades discretas: hoy mismo un producto en **latas** muestra consumos de **0.65–0.75 latas/semana**, y uno en **unidad** muestra **0.05–0.07 unidades/semana**. El reclamo original ("¿cómo alguien consume 0.7 de una lata?") sigue siendo válido a nivel de dato, solo que ahora está mejor etiquetado. **Hallazgo adicional (6 jul):** `stock_fin_semana` es **negativo en 61.8% de las filas** (1895 de 3068; 57 de 59 productos reales, hasta −343.87 en el peor caso) — un stock físico no puede ser negativo. Es la misma familia de problema, más grave de lo que parecía. Detalle y evidencia en `notebooks/modelo_cantidad_inventario.ipynb` (sección 9). |
| 2 | Categorías grandes de negocio | ✅ **Resuelto** | 11 categorías generales (`categoria_general`): Abarrotes, Limpieza e higiene, Menestras, Cereales y farináceos, Bebidas e infusiones, Cárnicos y pescados, Snacks y dulces, **Avícolas** (68 Aves + 17 Huevos, ya fusionados), Conservas, Lácteos, Otros. |
| 3 | Separar código interno de nombre visible | ✅ **Resuelto** | `dashboard.py` muestra `nombre_producto`; el código (`codigo_articulo`) queda solo como apoyo de búsqueda (tab Inventario, buscador). |
| 4 | Revisar si los códigos representan productos reales | ✅ **Resuelto** | Columna `es_producto`: 602 de 1678 códigos son productos reales con nombre; el resto (1076, incluyendo todos los `SINT####`) quedan marcados como registros técnicos y **no aparecen** en la vista operativa. |
| 5 | Homogeneizar el nivel de detalle (lenguaje de papel/Excel) | ✅ **Resuelto** | Vista "Personal del albergue" separada de la vista técnica; sin jerga de variables crudas en la vista de usuario. |
| 6 | Corregir notebook que mezcla dominios | ✅ **Resuelto** | `EDA_INVENTARIO.ipynb` fue regenerado (`scripts/generar_eda_inventario.py`) para usar los datos reales del almacén ALDIMI en vez del dataset genérico anterior. |
| 7 | Lenguaje operativo en la app | ✅ **Resuelto** | La app usa "reponer", "stock actual", "unidad", "categoría general", "atender" en vez de nombres de variables técnicas. |

**6 de 7 puntos de Jairo están genuinamente resueltos** — vale la pena que el equipo se dé crédito por esto. El punto 1 (unidades) es el único que sigue abierto de fondo, y es justo el que motivó todo el reclamo original, así que conviene priorizarlo o al menos tener una explicación clara para la sustentación (ver sección 4).

---

## 2. Lo que se acordó en el chat interno (4–5 jul) — estado real

| Pedido (quién) | Estado | Evidencia / detalle |
|---|---|---|
| Modelo de riesgo de pacientes binario (Alto/Bajo + "Revisión"), Iván, 94% acc. | ✅ **Implementado** | `dashboard.py` carga `models/risk_model_rf_binary.joblib` (no `models_pacientes.pkl`, que quedó huérfano). Bandas por score: Bajo &lt;0.35, Revisión 0.35–0.67, Alto &gt;0.67. |
| Vista simple para usuario común + vista técnica/detallada | ✅ **Implementado** | Selector en sidebar: "Personal del albergue" vs. "Desarrollador / Analista", con pestañas separadas en cada una. |
| Predecir **cantidad de productos** en vez de riesgo de quiebre (propuesta de Leonel, 4 jul) | 🔧 **Prototipo listo, falta que el equipo lo revise** | Se construyó `notebooks/modelo_cantidad_inventario.ipynb` (+ `scripts/generar_modelo_cantidad_inventario.py` + `models/modelo_cantidad_inventario.pkl`): regresión de consumo a 7/14 días (Regresión Lineal / Random Forest, comparados también contra un baseline ingenuo y XGBoost, con split temporal — no aleatorio), convertida en `cantidad_a_pedir` redondeada por tipo de unidad. **Con honestidad:** el baseline ingenuo (persistencia del promedio móvil) es muy competitivo, y el modelo *no* generaliza de forma confiable más allá de 40–60 familias (rango histórico) — para la proyección a 100 familias del enunciado se documentó una regla de escalamiento per-cápita transparente en vez de confiar en la extrapolación del modelo. Esto **no reemplaza** el modelo de clasificación de Leonel si él ya tiene algo propio — es un punto de partida para que el equipo decida y, si se usa, falta integrarlo al dashboard (no se tocó `dashboard.py` a propósito, para no chocar con el trabajo en curso). |
| Excel tipo "lista de compras" | ⚠️ **Parcial** | Hay botón "Descargar lista (CSV)" en la pestaña Inventario (`dashboard.py` línea ~332). Cumple la función, pero es CSV, no Excel con formato. Confirmar si basta o si querían algo más presentable para el personal de ALDIMI. |
| Que el dashboard permita **atender** alertas (marcar como repuesto) | 🔧 **Prototipo listo** | `dashboard_operativo.py` ya permite registrar ingresos de producto (ver sección 2bis); falta que el equipo decida si esto reemplaza o se suma al "Descargar lista (CSV)" de `dashboard.py`. |
| Que el dashboard permita **actualizar datos** | 🔧 **Prototipo listo (vía CSV, no `aldimi_core.db`)** | `dashboard_operativo.py` ya tiene formularios de entrada dentro del propio Streamlit (ver sección 2bis). Nota técnica: en este entorno de trabajo, `aldimi_core.db` (SQLite) dio errores de I/O que impidieron leer/escribir de forma confiable, así que el prototipo usa los CSV de `data/processed/` directamente — más simple de auditar para un equipo chico. Si el equipo prefiere todo centralizado en la BD, es un cambio puntual en las funciones `guardar_*`. |
| Pacientes: editar variables de un paciente **existente** para ver si baja su riesgo | ✅ **Prototipo listo** | `dashboard_operativo.py` ya permite cargar un paciente existente, separa sus variables clínicas fijas (solo lectura) de las accionables (editables), y muestra el cambio de prioridad en texto plano. Validado con los 45 pacientes reales — ver sección 2bis para el detalle. |
| Quitar la métrica "Familias #" hardcodeada | ⚠️ **Corregida, no eliminada** | Ya no está hardcodeada: `ctx["ocupacion_actual"]` se calcula desde `df_inv` real. Se mantiene como métrica ("Familias alojadas") porque calza con el reto del enunciado ("de 50 a 100 familias"). Fue una corrección razonable, pero no lo que se pidió literalmente (quitarla) — confirmar con el equipo que fue una decisión consciente. |
| Entrenar con los 600 pacientes existentes + 50–70 nuevos para prueba | ⚠️ **Aproximado, y a verificar** | Existe `aldimi_pacientes_sintetico_demo.csv` (600 filas) y `aldimi_pacientes_sintetico.csv` (45 filas, ligeramente bajo el rango 50–70 pedido). **Ojo:** el archivo de 600 filas **no está referenciado en ningún `.py`, `.ipynb` ni `.md` del repo** — existe en `data/processed/` pero nada lo usa. El notebook oficial (`notebooks/aldimi_analisis_modelado.ipynb`, celda 31) sigue cargando `aldimi_pacientes_sintetico.csv` (el de 45) y tratándolo como 3 clases (`Bajo/Medio/Alto`), igual que antes. El modelo binario (`risk_model_rf_binary.joblib`) no aparece generado en ningún notebook del repo — probablemente se entrenó en una sesión aparte (Iván mencionó trabajarlo con "Claude" fuera del notebook). Falta confirmar con qué dataset se entrenó realmente ese modelo y dejarlo documentado. |
| "Full UI/UX" | ✅ **Muy avanzado** | Dashboard nuevo, tema en `.streamlit/config.toml`, solo componentes nativos de Streamlit (sin CSS/HTML manual), arquitectura ordenada. Buen trabajo, pero ver el riesgo de la sección 0 (nada de esto está commiteado) y la limpieza pendiente (sección 4). |

---

### 2bis. Feedback más nuevo de Jairo (6 jul, transmitido por un compañero)

| Pedido | Estado | Evidencia / detalle |
|---|---|---|
| Consumo debe subir automáticamente si suben las familias, y recalcular cuándo se acaba el stock | ✅ **Prototipo listo** | `dashboard_operativo.py`: input numérico de "familias" que recalcula en vivo los días de cobertura de cada producto, usando una tasa de consumo histórica **por familia** (no un modelo de caja negra). Probado con datos reales: p. ej. Arverja partida pasa de "se acaba en 6 días" a "se acaba en 3 días" al subir de 50 a 100 familias — la relación es monotónica por construcción, no por suerte. |
| Cero tecnicismos en la vista del personal (nada de gráficos, scores, % de confianza, "accuracy", jerga de ML) | ✅ **Prototipo listo** | `dashboard_operativo.py` no importa ni usa librerías de gráficos; el modelo de riesgo de pacientes se sigue usando por dentro (`risk_model_rf_binary.joblib`) pero su score **nunca se muestra** — se traduce a una de 3 frases fijas ("🔴 Necesita atención pronto" / "🟡 Revisar cuando se pueda" / "🟢 Sin urgencia por ahora"). Las métricas técnicas siguen solo en `dashboard.py → vista_desarrollador()`, que no se tocó. |
| El dashboard debe permitir ver **e introducir** datos, tanto de inventario como de pacientes | ✅ **Prototipo listo** | Pestaña Almacén: formulario para registrar ingresos de productos (se guardan en `data/processed/movimientos_inventario.csv` y se reflejan al instante en el stock y la cobertura). Pestaña Pacientes: formulario para dar de alta un paciente nuevo y para editar uno existente (variables accionables: apoyo familiar, adherencia, acceso a medicamentos, estado nutricional, estado emocional), con el cambio de prioridad mostrado en texto plano. |

**Validado con datos reales antes de entregar** (no solo "corre sin errores"): sobre los 45 pacientes actuales, editar las variables accionables a sus valores más favorables cambia la banda de prioridad en 15 casos, y a los menos favorables en 21 casos (33 de 45 pacientes reaccionan en al menos una dirección). En los 12 restantes la banda no se mueve porque el modelo le da más peso a variables clínicas (etapa, infección) que el personal no puede modificar — es un resultado esperado, no un error, y coincide con lo ya documentado en el guion de video sobre importancia de variables.

**Nota:** es un prototipo aparte (`streamlit run dashboard_operativo.py`), pensado para que el equipo lo revise y decida si lo integra a `dashboard.py`, lo reemplaza, o toma de él solo las partes que le sirvan. No se modificó el dashboard principal.

---

## 3. Contraste con la rúbrica oficial del curso

El enunciado (`1ACC0057`, sección 4) es específico en dos puntos que vale la pena revisar con cuidado porque pueden pesar en la nota:

- **"Modelos de Regresión o Series Temporales"** para el frente de inventario (predicción de stock crítico + modelado de consumo según ocupación proyectada). Lo que existe hoy en `models_inventario.pkl` es **clasificación** (XGBoost/RF prediciendo `alerta_7_dias`/`alerta_14_dias` como sí/no), no una regresión de cantidad ni una serie temporal. Esto es exactamente lo que Leonel está tratando de resolver con su modelo de "cantidad de productos" — o sea que ese pendiente no es solo una mejora interna, es probablemente **un requisito textual del enunciado que todavía no se cumple**. Vale la pena tratarlo con esa prioridad.
- **"Nivel de Prioridad de Atención (Bajo/Medio/Alto)"** — el enunciado pide 3 niveles explícitamente. El equipo migró a un modelo **binario** (Alto/Bajo, con "Revisión" como zona ambigua no entrenada como clase). Es una decisión técnica defendible y bien razonada (la clase "Medio" confundía la separación de prioridad), pero se aleja de la letra del enunciado. Recomendación: tener lista una justificación corta y clara para la sustentación, y aprovechar que ya existe el expander "Ver etiqueta original del dataset" para mostrar la comparación 3 clases vs. binario si preguntan.

Sobre los 4 criterios de evaluación:

- **Funcionalidad técnica:** sólida en general (CV estratificada, `RandomizedSearchCV`, SHAP/permutation importance), con la salvedad del punto de regresión/series temporales de arriba.
- **Calidad del software (UX/UI + arquitectura limpia):** buen progreso — vista dual, tabs, sin hacks de CSS. Pendiente de limpieza menor (sección 4).
- **Metodología (SCRUM + CRISP-DM):** el uso de ramas de Git es buena práctica, pero se cae en la ejecución (nada commiteado, ver sección 0). La documentación CRISP-DM existe en el informe, pero puede haber quedado desactualizada frente a los últimos cambios de modelo (ver sección 4).
- **Impacto social / ODS:** cubierto en el informe y en el guion de video (ODS 2, 3, 10 bien hilados con la narrativa de ALDIMI 2.0).

Entregables de Hito 4 puntuales:

| Entregable pedido | Estado |
|---|---|
| Informe final completo | ⚠️ Existe (`Informe/TF_ALDIMI_v6.pdf`), pero la portada y la tabla de roles todavía tienen el placeholder `[Integrante N]` / `[U20XXXXXX]` sin completar (8 apariciones en `TF_ALDIMI_v6.tex`). |
| Dashboard final con alertas de stock y riesgo de pacientes | ✅ Implementado (ver secciones 1–2) |
| Video de impacto (flujo end-to-end con IA) | ❌ No grabado. El guion (`Informe/guion_video_y_exposicion_hito4.md`) está completo y listo para grabar. |
| Zip de entrega `/codigo /datos /docs` | ⚠️ **No existe ningún `.zip` en todo el repo** (se buscó explícitamente). Solo está la carpeta descomprimida `entrega/TF_1ASI404_3037_GRUPO_02/`, y esa copia además está desactualizada: su `dashboard.py` interno todavía carga el modelo viejo `models_pacientes.pkl` (multiclase), no el binario actual. Hay que regenerar todo el paquete de entrega desde el estado más reciente, no solo volver a comprimirlo. |
| Integración real con el módulo de IA (OCR → BD común) | ❌ Pendiente. La arquitectura está lista (`aldimi_core.db`, `integracion_bd.py`, fallback a CSV), pero el propio README señala como pendiente "coordinar con el grupo de IA la demo del OCR escribiendo en `aldimi_core.db`" — no hay evidencia de que se haya probado con datos reales del otro equipo. |

---

## 4. Detalles y bugs menores encontrados

- **Typo en datos:** fila `PAC-0201` en `aldimi_pacientes_sintetico.csv` tiene `nivel_riesgo = "MedioMedio"` en vez de `"Medio"`. Corrección de una celda.
- **`dashboard_legacy.py`** (835 líneas) sigue en la raíz y todavía carga el modelo viejo (`models_pacientes.pkl`, multiclase). Si no se usa, conviene archivarlo o eliminarlo para que no haya dos "dashboard" y se preste a confusión en la revisión del profesor.
- **`models/models_pacientes.pkl`** quedó huérfano para el dashboard vigente (`dashboard.py` usa `risk_model_rf_binary.joblib`); solo lo siguen cargando `dashboard_legacy.py` y la copia desactualizada dentro de `entrega/`. Decidir si se conserva como referencia documentada o se retira.
- **`analisis_arquitectura_ALDIMI.md`** está fechado 3 de julio y describe un estado anterior (modelo de pacientes multiclase como el vigente, y varias carpetas duplicadas — `data_aldimi/`, `data_nueva/`, `archivos/`, `Doc/` — que de hecho **ya no existen** en el repo, o sea que ese análisis ya cumplió su propósito de limpieza). Vale la pena actualizarlo o al menos anotar en él qué recomendaciones ya se aplicaron.
- **`guion_video_y_exposicion_hito4.md`** cita métricas y nombres de pantalla que ya no coinciden con la app actual: menciona "F1-Macro = 0.75 (pacientes)" (el modelo viejo) y páginas "Alertas de Inventario" / "Evaluación de Pacientes" que ahora se llaman distinto (tabs "Resumen Operativo" / "Inventario" / "Pacientes" dentro de cada modo). Actualizarlo antes de ensayar o grabar, para no presentar números que contradicen lo que se ve en vivo en el dashboard.
- **602 de 1678 códigos** del catálogo tienen nombre interpretable; el resto queda oculto de la vista operativa. Parece una simplificación intencional y razonable, pero confirmar con el equipo que no se está escondiendo algún producto real relevante.
- **`dashboard_operativo.py` (nuevo)** guarda los ingresos de inventario en `data/processed/movimientos_inventario.csv`, un archivo nuevo que no existía antes — no se creó todavía en el repo real (solo se probó contra copias temporales), así que no hay que buscarlo hasta que alguien corra el prototipo por primera vez.

---

## 5. Orden sugerido para lo que queda de tiempo

1. **Commitear y pushear todo ya** (sección 0). Sin esto, nada de lo demás importa si se pierde el trabajo.
2. **Revisar `dashboard_operativo.py`** (sección 2bis) — cubre el feedback más reciente y más explícito de Jairo (familias dinámicas, cero tecnicismos, ver+introducir datos). Decidir si se integra a `dashboard.py`, lo reemplaza, o se usa solo como referencia.
3. **Revisar el prototipo de modelo de cantidad** (`notebooks/modelo_cantidad_inventario.ipynb`) junto con Leonel: decidir si se usa, se combina con lo suyo, o se descarta — y si se usa, integrarlo al dashboard. Cubre el requisito textual del enunciado de "regresión o series temporales" que la clasificación binaria no cubría.
4. **Decidir qué hacer con el problema de unidades fraccionarias y de stock negativo** (61.8% de las filas con `stock_fin_semana` &lt; 0): o se corrige la generación del dato, o como mínimo se explica en el informe qué representa ese número, para no repetir el mismo comentario de Jairo en la sustentación.
5. **Completar nombres y códigos reales** en la portada de `TF_ALDIMI_v6.tex` y recompilar (`xelatex` ×2).
6. **Regenerar el ZIP de entrega** con el estado más reciente de código/datos/notebooks.
7. **Grabar el video de impacto** — el guion ya está listo, pero actualizarlo primero (métricas y nombres de pantalla, punto 4 de arriba).
8. **Ensayar con el equipo de IA** el flujo end-to-end real (OCR → BD común → Dashboard) al menos una vez antes de la sustentación.
9. **Limpieza final:** `dashboard_legacy.py`, `models_pacientes.pkl`, typo "MedioMedio", refresco de `analisis_arquitectura_ALDIMI.md`.

---

## 6. Para no perder de vista

La mayor parte del feedback de Jairo (6 de 7 puntos del 2 jul, y los 3 puntos nuevos del 6 jul) y buena parte de lo acordado internamente ya está resuelto o prototipado con calidad real: catálogo limpio, categorías de negocio, vista dual usuario/técnico, modelo binario de pacientes, arquitectura de BD compartida con el equipo de IA, y ahora un dashboard operativo sin tecnicismos con consumo dinámico por familias. Lo que queda pendiente es puntual y accionable con los días que quedan — el riesgo más grande ahora mismo no es de contenido, sino de **que este avance no está ni commiteado ni pusheado**.
