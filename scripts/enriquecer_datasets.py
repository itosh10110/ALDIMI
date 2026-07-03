# -*- coding: utf-8 -*-
"""
Enriquece los datasets procesados con el catálogo de productos:
nombre visible, presentación, unidad de medida estándar y categoría general.

Ejecutar DESPUÉS de scripts/build_catalogo.py.
Modifica in-place:
  - data/processed/aldimi_dataset_completo.csv
  - data/processed/aldimi_dataset_semanal.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

COLS_CATALOGO = [
    "codigo_articulo",
    "nombre_producto",
    "presentacion",
    "unidad_medida",
    "categoria_general",
    "es_producto",
]


def main() -> None:
    catalogo = pd.read_csv(PROCESSED / "catalogo_productos.csv")[COLS_CATALOGO]

    for nombre in ("aldimi_dataset_completo.csv", "aldimi_dataset_semanal.csv"):
        path = PROCESSED / nombre
        df = pd.read_csv(path)
        # quitar columnas del catálogo si ya existían (re-ejecución idempotente)
        df = df.drop(columns=[c for c in COLS_CATALOGO[1:] if c in df.columns])
        df = df.merge(catalogo, on="codigo_articulo", how="left")
        df["es_producto"] = df["es_producto"].fillna(False).astype(bool)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  ✓ {nombre}: {len(df)} filas, columnas nuevas añadidas")

    print("Datasets enriquecidos con nombre, unidad y categoría general.")


if __name__ == "__main__":
    main()
