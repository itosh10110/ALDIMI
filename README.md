
# ALDIMI Predict — Sistema Predictivo de Gestión
**Machine Learning 1ACC0057 · UPC · Grupo 2 · Hito 3**

> Sistema de predicción de desabastecimiento de inventario y clasificación de riesgo de pacientes para el Albergue Divina Misericordia (ALDIMI).

---

## Estructura del proyecto

```
TB1/
├── dashboard.py              ← App principal (Streamlit)
├── lanzar_demo.py            ← Demo pública con ngrok
├── requirements.txt          ← Dependencias Python
│
├── models/                   ← Modelos entrenados (PKL)
│   ├── models_inventario.pkl   (XGBoost 7d + RF 14d)
│   └── models_pacientes.pkl    (XGBoost multiclase)
│
├── data/
│   ├── raw/                  ← Datos originales de ALDIMI
│   └── processed/            ← CSVs listos para el dashboard
│       ├── aldimi_dataset_semanal.csv
│       ├── aldimi_dataset_completo.csv
│       └── aldimi_pacientes_sintetico.csv
│
├── notebooks/
│   └── aldimi_analisis_modelado.ipynb   ← Entrenamiento completo
│
├── figures/
│   ├── hito2/               ← Figuras EDA y baseline (g1–g12)
│   └── hito3/               ← Figuras modelado avanzado (fig13–fig17)
│
├── Informe/
│   ├── TB1_ALDIMI_v5.pdf    ← Informe Hito 3 (ENTREGABLE)
│   ├── TB1_ALDIMI_v5.tex    ← Fuente LaTeX
│   ├── TB1_ALDIMI_v5.docx   ← Versión Word
│   ├── guion_exposicion_hito2.md
│   └── versiones/           ← Versiones previas del informe
│
└── archivos/                ← RAR de respaldo
```

---

## Instalación rápida

### 1. Crear entorno virtual (solo la primera vez)

```bash
python -m venv .venv
```

### 2. Activar el entorno

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Windows (CMD):** `.venv\Scripts\activate.bat`

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Ejecutar el dashboard localmente

```bash
# Con el entorno activado:
streamlit run dashboard.py
```

Abre automáticamente en `http://localhost:8501`

---

## Compartir el dashboard con el equipo (demo pública)

```bash
# Con el entorno activado:
python lanzar_demo.py
```

Te pedirá tu token de [ngrok](https://ngrok.com) la primera vez (registro gratuito).  
Genera un enlace público tipo `https://xxxx.ngrok-free.app` que el equipo puede abrir desde cualquier dispositivo.

---

## Recompilar el informe PDF

Requiere **XeLaTeX** instalado (incluido en TeX Live o MiKTeX).

```bash
cd Informe
xelatex TB1_ALDIMI_v5.tex
xelatex TB1_ALDIMI_v5.tex   # segunda pasada para TOC
```

---

## Modelos del Hito 3

| Tarea | Modelo seleccionado | F1 | AUC |
|---|---|---|---|
| Inventario 7 días | XGBoost | 0.9865 | 0.9982 |
| Inventario 14 días | Random Forest | 0.9721 | 0.9901 |
| Riesgo pacientes | XGBoost | 0.7500 (Macro) | 0.8710 |

Los modelos fueron entrenados con `RandomizedSearchCV` (50–60 iteraciones) y `StratifiedKFold` (5 folds). Ver el notebook en `notebooks/` para reproducir el entrenamiento completo.

---

## ODS alineados
- **ODS 2** — Hambre Cero: anticipa quiebres de stock en dieta oncológica
- **ODS 3** — Salud y Bienestar: clasifica riesgo clínico de pacientes pediátricos
- **ODS 10** — Reducción de Desigualdades: herramienta gratuita para ONGs vulnerables

---

*ML 1ACC0057 · UPC · Junio 2026*
# ALDIMI

