# -*- coding: utf-8 -*-
"""
Construye el catálogo maestro de productos ALDIMI.

Responde a las observaciones del profesor:
- Separa código interno (codigo_articulo) del nombre visible (nombre_producto).
- Define una unidad de medida estándar por tipo de producto.
- Agrupa las 25 categorías específicas en categorías generales de negocio.
- Marca qué códigos son productos reales y cuáles son registros técnicos
  (sintéticos SINT#### o filas sin descripción en el Excel original).

Salida: data/processed/catalogo_productos.csv
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "data" / "raw" / "ALMAcen upc.xlsx"
COMPLETO = ROOT / "data" / "processed" / "aldimi_dataset_completo.csv"
OUT = ROOT / "data" / "processed" / "catalogo_productos.csv"

# ---------------------------------------------------------------- reglas de negocio
# Unidad de medida estándar por tipo de producto (categoría original del almacén)
UNIDAD_POR_CATEGORIA = {
    "Menestras": "kg",
    "Azucar": "kg",
    "Harina": "kg",
    "Cereal": "kg",
    "Fideos": "kg",
    "Frutos secos": "kg",
    "Aves": "kg",
    "Vacuno": "kg",
    "Cerdo": "kg",
    "Cordero": "kg",
    "Pescado": "kg",
    "Aceite": "litro",
    "Conserva": "lata",
    "Lacteos": "lata",
    "Infusion": "caja",
    "Snacks": "paquete",
    "Embutidos": "paquete",
    "Huevos": "unidad",
    "Limpieza": "unidad",
    "Cuidado Personal": "unidad",
    "Condimentos, Salsas, cremas": "unidad",
    "Mermelada, mantequilla": "unidad",
    "Instantaneo": "unidad",
    "Gelatina, postres": "unidad",
    "Otros": "unidad",
}

# Agrupación en categorías generales de negocio
CATEGORIA_GENERAL = {
    "Aves": "Avícolas",
    "Huevos": "Avícolas",
    "Vacuno": "Cárnicos y pescados",
    "Cerdo": "Cárnicos y pescados",
    "Cordero": "Cárnicos y pescados",
    "Embutidos": "Cárnicos y pescados",
    "Pescado": "Cárnicos y pescados",
    "Menestras": "Menestras",
    "Lacteos": "Lácteos",
    "Cereal": "Cereales y farináceos",
    "Harina": "Cereales y farináceos",
    "Fideos": "Cereales y farináceos",
    "Conserva": "Conservas",
    "Aceite": "Abarrotes",
    "Azucar": "Abarrotes",
    "Condimentos, Salsas, cremas": "Abarrotes",
    "Mermelada, mantequilla": "Abarrotes",
    "Infusion": "Bebidas e infusiones",
    "Instantaneo": "Bebidas e infusiones",
    "Snacks": "Snacks y dulces",
    "Gelatina, postres": "Snacks y dulces",
    "Frutos secos": "Snacks y dulces",
    "Limpieza": "Limpieza e higiene",
    "Cuidado Personal": "Limpieza e higiene",
    "Otros": "Otros",
}

# Detección de presentación dentro del nombre (500 gr, 1 kg, 3 lt, 25 und, lata…)
RE_PRESENTACION = re.compile(
    r"\b(\d+[.,]?\d*)\s*(gr|g|kg|kl|ml|lt|l|und|u)\b\.?", re.IGNORECASE
)


def limpiar_nombre(detalle: str) -> str:
    """Nombre visible: texto original sin la presentación, con espacios normalizados."""
    s = unicodedata.normalize("NFKC", str(detalle)).strip()
    s = re.sub(r"\s+", " ", s)
    return s[:1].upper() + s[1:] if s else s


def extraer_presentacion(detalle: str) -> str:
    m = RE_PRESENTACION.search(str(detalle))
    if not m:
        for envase in ("lata", "caja", "bandeja", "bolsa", "frasco"):
            if envase in str(detalle).lower():
                return envase
        return ""
    cant, un = m.group(1), m.group(2).lower()
    un = {"g": "gr", "kl": "kg", "l": "lt", "u": "und"}.get(un, un)
    return f"{cant} {un}"


def main() -> None:
    raw = pd.read_excel(RAW_XLSX, sheet_name="Existencias", header=1)
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw[raw["Código"].notna()].drop_duplicates(subset="Código")
    raw = raw.rename(
        columns={"Código": "codigo_articulo", "Nombre del artículo2": "categoria"}
    )

    completo = pd.read_csv(COMPLETO)
    base = completo[["codigo_articulo", "categoria", "origen"]].drop_duplicates(
        subset="codigo_articulo"
    )

    cat = base.merge(
        raw[["codigo_articulo", "Detalle"]], on="codigo_articulo", how="left"
    )

    cat["nombre_producto"] = cat["Detalle"].apply(
        lambda d: limpiar_nombre(d) if pd.notna(d) else ""
    )
    cat["presentacion"] = cat["Detalle"].apply(
        lambda d: extraer_presentacion(d) if pd.notna(d) else ""
    )
    cat["unidad_medida"] = cat["categoria"].map(UNIDAD_POR_CATEGORIA).fillna("unidad")
    cat["categoria_general"] = cat["categoria"].map(CATEGORIA_GENERAL).fillna("Otros")

    # Un código es un PRODUCTO visible para el usuario operativo solo si:
    #  - proviene del almacén real (no SINT####) y
    #  - tiene descripción en el Excel original.
    # El resto son registros técnicos: se conservan para los modelos,
    # pero no se muestran en la app.
    cat["es_producto"] = (cat["origen"] == "real") & (cat["nombre_producto"] != "")
    cat.loc[~cat["es_producto"] & (cat["nombre_producto"] == ""), "nombre_producto"] = (
        "Registro técnico (sin descripción)"
    )

    cols = [
        "codigo_articulo",
        "nombre_producto",
        "presentacion",
        "unidad_medida",
        "categoria",
        "categoria_general",
        "origen",
        "es_producto",
    ]
    cat[cols].to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"Catálogo generado: {OUT}")
    print(f"  Total códigos:        {len(cat)}")
    print(f"  Productos visibles:   {cat['es_producto'].sum()}")
    print(f"  Registros técnicos:   {(~cat['es_producto']).sum()}")
    print("\nProductos por categoría general (solo visibles):")
    print(cat[cat["es_producto"]]["categoria_general"].value_counts().to_string())


if __name__ == "__main__":
    main()
