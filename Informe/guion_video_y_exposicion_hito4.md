# Guion — Video de Impacto y Exposición Final (Hito 4)

**ALDIMI Predict · ML 1ACC0057 · Grupo 2 · Julio 2026**

---

## Parte A — Video de Impacto (3–4 min)

> Grabar con OBS/Loom sobre el dashboard corriendo en local. Tener el video listo
> ANTES de la exposición como respaldo ante fallas técnicas (requisito del enunciado).

### Escena 1 — El problema (30 s)
- Imagen/foto de ALDIMI + texto: "50 → 100 familias. Procesos en papel."
- Narración: "ALDIMI alberga a niños con cáncer en extrema pobreza. Su expansión al
  doble de capacidad es imposible de sostener con kardex manual y priorización a ojo."

### Escena 2 — Flujo end-to-end (90 s) ⭐ *lo que pide el enunciado*
1. **Captura (módulo IA):** mostrar el OCR/chatbot del equipo de IA registrando una
   ficha de ingreso. *(Coordinar con el grupo de IA; si no está disponible, mostrar
   la inserción del registro con `integracion_bd.py`.)*
2. **BD común:** terminal con `python integracion_bd.py --verificar` mostrando las
   tablas `pacientes`, `inventario_semanal` y la bitácora `sync_log`.
3. **Predicción (módulo ML):** el dashboard con el sidebar mostrando
   "🔗 Fuente de datos: BD común (aldimi_core.db)".

### Escena 3 — Demo de valor (60 s)
- Página **Alertas de Inventario**: consultar un artículo con stock bajo →
  quiebre probable a 7 días con ~99% de probabilidad → "reposición urgente".
- Página **Evaluación de Pacientes**: caso etapa IV + desnutrición + infección →
  riesgo ALTO con factores explicados.

### Escena 4 — Cierre e impacto (30 s)
- Métricas en pantalla: F1 = 0.9865 (stock 7d) · F1-Macro = 0.75 (pacientes).
- "Menos compras de emergencia, priorización temprana de pacientes.
  ODS 2, 3 y 10. Replicable a 200+ organizaciones en el Perú."

---

## Parte B — Exposición final (15 min + 5 Q&A)

| # | Sección | Tiempo | Expositor | Contenido clave |
|---|---------|--------|-----------|-----------------|
| 1 | Pitch de impacto | 2 min | — | Problema ALDIMI 2.0, ODS 2/3/10 |
| 2 | CRISP-DM recorrido | 2 min | Data Scientist | Fases 1–3: datos, EDA, features |
| 3 | Modelado y evaluación | 3 min | ML Engineer | XGBoost vs RF, CV 5-fold, SHAP |
| 4 | **Demo en vivo** | 4 min | MLOps | Dashboard + BD común (¡ensayar 2 veces!) |
| 5 | Integración ecosistema | 2 min | Data Architect | BD común, contrato con IA, sync_log |
| 6 | Métricas y ética | 1.5 min | — | Por qué los resultados son aceptables para ALDIMI; sesgos y human-in-the-loop |
| 7 | Cierre | 0.5 min | — | Impacto cuantificado y siguientes pasos |

### Preguntas probables del jurado (preparar a TODOS los integrantes)
1. **¿Por qué F1 y no accuracy?** → Clases desbalanceadas; accuracy engaña con la
   clase mayoritaria. F1-Macro pondera igual las 3 clases de riesgo.
2. **¿Por qué XGBoost a 7d pero Random Forest a 14d?** → Se seleccionó el mejor F1
   en test por horizonte; a 14d RF generalizó mejor (F1 0.9721 vs 0.9650).
3. **¿El 0.75 de F1-Macro en pacientes es suficiente?** → Sí como sistema de
   priorización con confirmación humana: +23.8% sobre baseline y AUC-Macro 0.92;
   nunca decide solo, prioriza la cola de atención.
4. **¿Cómo evitan sesgos contra familias rurales/pobres?** → SHAP muestra dominancia
   de variables clínicas; auditoría periódica por procedencia; human-in-the-loop.
5. **¿Qué pasa si la ocupación sube de 100 familias?** → Fuera del rango de
   entrenamiento → política de reentrenamiento y modo consultivo.
6. **¿Cómo se integran con IA?** → BD común SQLite con contrato de interfaz:
   IA inserta, ML lee; bitácora sync_log; fallback a CSV.
7. **¿Datos reales o sintéticos?** → Inventario real (kardex); pacientes sintéticos
   derivados de fichas reales, por protección de datos de menores (Ley 29733).

### Checklist previo a la sustentación
- [ ] Video de respaldo grabado y descargado (no depender de internet)
- [ ] Dashboard probado en la laptop de exposición (`streamlit run dashboard.py`)
- [ ] `data/aldimi_core.db` presente (ejecutar `python integracion_bd.py` si falta)
- [ ] Slides exportadas a PDF (por si falla PowerPoint)
- [ ] Todos los integrantes conocen las 7 respuestas de arriba
