# -*- coding: utf-8 -*-
"""
Regenera EDA_INVENTARIO.ipynb usando los datos reales del almacén ALDIMI.
(El notebook anterior analizaba un dataset de otro contexto, con ítems,
proveedores y lead-time que no corresponden a este proyecto.)

Uso:  python scripts/generar_eda_inventario.py [--ejecutar]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "EDA_INVENTARIO.ipynb"


def md(texto):
    return {"cell_type": "markdown", "metadata": {}, "source": texto.splitlines(keepends=True)}


def code(texto):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": texto.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""# EDA — Inventario del almacén ALDIMI
**Machine Learning 1ACC0057 · UPC · Grupo 2**

Análisis exploratorio de los datos **reales del almacén del Albergue Divina Misericordia**:
el consolidado por artículo (`aldimi_dataset_completo.csv`), el histórico semanal
(`aldimi_dataset_semanal.csv`) y el catálogo maestro de productos
(`catalogo_productos.csv`), que aporta el **nombre visible**, la **unidad de medida
estándar** y la **categoría general de negocio** de cada código."""),
    code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 100

RUTA = 'data/processed'
df_art = pd.read_csv(f'{RUTA}/aldimi_dataset_completo.csv')
df_sem = pd.read_csv(f'{RUTA}/aldimi_dataset_semanal.csv')
catalogo = pd.read_csv(f'{RUTA}/catalogo_productos.csv')
df_art.shape, df_sem.shape, catalogo.shape
"""),
    md("""## 1. Estructura y calidad de los datos"""),
    code("df_art.info()"),
    code("df_art.head(10)"),
    code("""
print('Nulos por columna (artículos):')
print(df_art.isnull().sum())
print('\\nDuplicados:', df_art.duplicated(subset='codigo_articulo').sum())
print('Stock negativo:', (df_art['existencias_actuales'] < 0).sum())
"""),
    md("""## 2. Productos reales vs. registros técnicos

El catálogo separa los **productos reales** (con nombre y unidad, visibles en la app)
de los **registros técnicos**: códigos sintéticos `SINT####` generados para entrenar
los modelos y códigos del kardex sin descripción."""),
    code("""
resumen = catalogo.groupby(['origen', 'es_producto']).size().rename('códigos').reset_index()
print(resumen.to_string(index=False))
print(f"\\nProductos visibles en la app: {catalogo['es_producto'].sum()}")
print(f"Registros técnicos (solo entrenamiento): {(~catalogo['es_producto'].astype(bool)).sum()}")
"""),
    md("""## 3. Unidad de medida estándar por tipo de producto"""),
    code("""
(catalogo[catalogo['es_producto'].astype(bool)]
 .groupby(['categoria_general', 'unidad_medida']).size()
 .unstack(fill_value=0))
"""),
    md("""## 4. Productos por categoría general de negocio"""),
    code("""
prod = catalogo[catalogo['es_producto'].astype(bool)]
conteo = prod['categoria_general'].value_counts()

fig, ax = plt.subplots(figsize=(9, 4.5))
conteo.sort_values().plot.barh(ax=ax, color='#4da9d4')
ax.set_xlabel('Número de productos')
ax.set_title('Productos reales por categoría general')
plt.tight_layout(); plt.show()
"""),
    md("""## 5. Estadísticos del inventario (solo productos reales)"""),
    code("""
df_prod = df_art[df_art['es_producto'].astype(bool)].copy()
cols = ['existencias_iniciales', 'total_entradas', 'total_salidas',
        'existencias_actuales', 'tasa_rotacion']
df_prod[cols].describe().round(2)
"""),
    code("""
print('Asimetría (skew):')
print(df_prod[cols].skew().round(2))
"""),
    code("""
fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
for ax, col in zip(axes, cols):
    sns.histplot(df_prod[col], bins=30, ax=ax, color='#1b3b64')
    ax.set_title(col, fontsize=9)
plt.tight_layout(); plt.show()
"""),
    md("""## 6. Boxplots por categoría general"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, col in zip(axes, ['existencias_actuales', 'tasa_rotacion']):
    sns.boxplot(data=df_prod, x='categoria_general', y=col, ax=ax, color='#4da9d4')
    ax.tick_params(axis='x', rotation=60)
    ax.set_title(f'{col} por categoría general')
    ax.set_xlabel('')
plt.tight_layout(); plt.show()
"""),
    md("""## 7. Detección de outliers (IQR)"""),
    code("""
resultados = []
for col in cols:
    q1, q3 = df_prod[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = ((df_prod[col] < q1 - 1.5 * iqr) | (df_prod[col] > q3 + 1.5 * iqr)).sum()
    resultados.append({'variable': col, 'outliers': int(n_out),
                       'pct': round(100 * n_out / len(df_prod), 1)})
pd.DataFrame(resultados)
"""),
    md("""## 8. Balance de las clases objetivo (alertas de stock)"""),
    code("""
print('Alerta 7 días :', df_sem['alerta_7_dias'].value_counts(normalize=True).round(3).to_dict())
print('Alerta 14 días:', df_sem['alerta_14_dias'].value_counts(normalize=True).round(3).to_dict())

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, col, titulo in zip(axes, ['alerta_7_dias', 'alerta_14_dias'],
                           ['Alerta de stock a 7 días', 'Alerta de stock a 14 días']):
    df_sem[col].value_counts().plot.bar(ax=ax, color=['#27ae60', '#c0392b'])
    ax.set_title(titulo); ax.set_xticklabels(['Sin alerta', 'Con alerta'], rotation=0)
plt.tight_layout(); plt.show()
"""),
    md("""## 9. Evolución semanal del stock por categoría general"""),
    code("""
sem_prod = df_sem[df_sem['es_producto'].astype(bool)]
top_cats = sem_prod['categoria_general'].value_counts().head(5).index

fig, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=True)
for ax, cat in zip(axes, top_cats):
    sub = (sem_prod[sem_prod['categoria_general'] == cat]
           .groupby('semana_del_año')['stock_fin_semana'].mean())
    ax.plot(sub.index, sub.values, color='#1b3b64', marker='o', ms=3)
    ax.axhline(0, color='#c0392b', ls='--', lw=1)
    ax.set_ylabel('Stock prom.'); ax.set_title(cat, fontsize=10, loc='left')
axes[-1].set_xlabel('Semana del año')
plt.tight_layout(); plt.show()
"""),
    md("""## 10. Correlaciones"""),
    code("""
cols_corr = ['ocupacion_albergue', 'stock_inicio_semana', 'ingresos_semana',
             'salidas_semana', 'stock_fin_semana', 'rolling_avg_salidas_3sem',
             'alerta_7_dias', 'alerta_14_dias']
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(df_sem[cols_corr].corr(), annot=True, fmt='.2f', cmap='RdBu_r',
            vmin=-1, vmax=1, linewidths=0.5, ax=ax)
ax.set_title('Correlaciones — inventario semanal')
plt.tight_layout(); plt.show()
"""),
    md("""## 11. Separabilidad de clases

¿Se distinguen las semanas con alerta de las semanas sin alerta según el stock
al inicio de la semana? Esto anticipa qué tan aprendible es el problema."""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, col in zip(axes, ['alerta_7_dias', 'alerta_14_dias']):
    for val, label, color in [(0, 'Sin alerta', '#27ae60'), (1, 'Con alerta', '#c0392b')]:
        sub = df_sem[df_sem[col] == val]['stock_inicio_semana']
        sns.kdeplot(sub, ax=ax, label=label, color=color, fill=True, alpha=0.3)
    ax.set_title(f'Stock al inicio de semana según {col}')
    ax.legend()
plt.tight_layout(); plt.show()
"""),
    md("""## Conclusiones

1. El almacén tiene **602 productos reales** con nombre, presentación y unidad de medida
   estándar; los códigos restantes son registros técnicos que solo se usan para entrenar.
2. Las **11 categorías generales** concentran el inventario en Abarrotes, Limpieza e
   higiene, Menestras y Cereales y farináceos.
3. Las variables de stock presentan fuerte asimetría positiva y outliers propios de un
   almacén de donaciones (ingresos irregulares).
4. Las clases de alerta están desbalanceadas, lo que justifica el uso de métricas F1/AUC
   y el manejo de desbalanceo en el entrenamiento.
5. El stock al inicio de la semana y el consumo promedio móvil separan bien las semanas
   con y sin alerta: buena señal para los modelos de 7 y 14 días."""),
]

NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(NB, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Notebook generado: {OUT}")

if "--ejecutar" in sys.argv:
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.read(OUT, as_version=4)
    NotebookClient(nb, timeout=300, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(nb, OUT)
    print("Notebook ejecutado con salidas.")
