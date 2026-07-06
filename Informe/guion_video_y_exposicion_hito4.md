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
- Modo **"Personal del albergue" → pestaña Inventario**: consultar un artículo con
  stock bajo → quiebre probable a 7 días con ~99% de probabilidad → "reposición urgente".
- Modo **"Personal del albergue" → pestaña Pacientes**: caso etapa IV + desnutrición +
  infección → banda de riesgo **Alto** con factores explicados (score del modelo binario).

### Escena 4 — Cierre e impacto (30 s)
- Métricas en pantalla: F1 = 0.9865 · AUC = 0.998 (stock 7d, XGBoost) · Accuracy = 94.8%
  · F1 = 0.95 · AUC = 0.99 (pacientes, RF binario Alto/Bajo, validado con Stratified
  5-fold CV sobre 405 casos). *(Actualizar aquí si Leonel entrega el nuevo modelo de
  cantidad para inventario antes de grabar — reemplaza F1/AUC por MAE/RMSE/R².)*
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
1. **¿Por qué F1 y no solo accuracy?** → Clases desbalanceadas; accuracy engaña con
   la clase mayoritaria. En pacientes reportamos ambas: accuracy 94.8% y F1 0.95
   (CV estratificada 5-fold), porque además el dataset binario quedó balanceado.
2. **¿Por qué XGBoost a 7d pero Random Forest a 14d?** → Se seleccionó el mejor F1
   en test por horizonte; a 14d RF generalizó mejor (F1 0.9721 vs 0.9650).
3. **¿Por qué pasaron de 3 clases (Bajo/Medio/Alto) a un modelo binario?** → El
   enunciado pide 3 niveles, pero la clase "Medio" no aportaba una frontera de
   decisión clara para priorizar atención (el propio profesor lo señaló: la etapa
   del cáncer por sí sola no ordena el riesgo linealmente). El equipo migró a
   Alto/Bajo + una banda de "Revisión" (score 0.35–0.67) para casos ambiguos, y
   subió de F1-Macro 0.75 a accuracy 94.8% / F1 0.95 (Stratified 5-fold CV, RF).
   La etiqueta original de 3 clases se conserva visible en el dashboard como
   referencia, no como lo que decide la prioridad.
4. **¿Cómo evitan sesgos contra familias rurales/pobres?** → Importancia por
   permutación muestra dominancia de variables clínicas (etapa, estado nutricional,
   infección) sobre variables socioeconómicas; auditoría periódica por procedencia;
   human-in-the-loop.
5. **¿Qué pasa si la ocupación sube de 100 familias?** → Fuera del rango de
   entrenamiento → política de reentrenamiento y modo consultivo.
6. **¿Cómo se integran con IA?** → BD común SQLite con contrato de interfaz:
   IA inserta, ML lee; bitácora sync_log; fallback a CSV.
7. **¿Datos reales o sintéticos?** → Inventario real (kardex); pacientes sintéticos
   derivados de fichas reales, por protección de datos de menores (Ley 29733).
8. **¿Por qué el inventario predice "riesgo de quiebre" y no cantidad a comprar?**
   *(pregunta esperada tras el pivote de Leonel — completar esta respuesta antes de
   la sustentación según el estado del nuevo modelo de regresión).*

### Checklist previo a la sustentación
- [ ] Video de respaldo grabado y descargado (no depender de internet)
- [ ] Dashboard probado en la laptop de exposición (`streamlit run dashboard.py`)
- [ ] `data/aldimi_core.db` presente (ejecutar `python integracion_bd.py` si falta)
- [ ] Slides exportadas a PDF (por si falla PowerPoint)
- [ ] Todos los integrantes conocen las 7 respuestas de arriba
