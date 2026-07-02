# ============================================================
#  ALDIMI Predict — Integración con la Base de Datos común
#  Machine Learning 1ACC0057 · UPC · Grupo 2 · Hito 4
# ------------------------------------------------------------
#  La BD común (aldimi_core.db, SQLite) es el punto de
#  confluencia con el módulo de IA (1ASI0404):
#    · El módulo de IA (OCR/chatbot) INSERTA registros en las
#      tablas `pacientes` e `inventario_semanal`.
#    · El módulo de ML (este proyecto) LEE esas tablas para
#      predecir y reentrenar.
#
#  Uso:
#    python integracion_bd.py            → crea/actualiza la BD desde los CSV
#    python integracion_bd.py --verificar → muestra el esquema y conteos
#
#  El dashboard (dashboard.py) lee automáticamente de la BD si
#  existe data/aldimi_core.db; si no, usa los CSV procesados.
# ============================================================
import sqlite3
import os
import sys
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, 'data', 'aldimi_core.db')
PROCESSED = os.path.join(BASE, 'data', 'processed')

TABLAS = {
    'inventario_semanal': 'aldimi_dataset_semanal.csv',
    'articulos':          'aldimi_dataset_completo.csv',
    'pacientes':          'aldimi_pacientes_sintetico.csv',
}


def crear_db(db_path: str = DB_PATH) -> None:
    """Crea/actualiza la BD común cargando los datasets procesados.

    En producción, las tablas `pacientes` e `inventario_semanal` las
    alimenta el módulo de IA (OCR de fichas y kardex); este script
    inicializa la BD con los datos históricos del albergue.
    """
    con = sqlite3.connect(db_path)
    for tabla, csv in TABLAS.items():
        df = pd.read_csv(os.path.join(PROCESSED, csv))
        df.to_sql(tabla, con, if_exists='replace', index=False)
        print(f"  ✓ {tabla:<20} {len(df):>5} filas ← {csv}")
    # Metadatos de sincronización (auditoría de la confluencia IA↔ML)
    con.execute("""CREATE TABLE IF NOT EXISTS sync_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fecha TEXT DEFAULT CURRENT_TIMESTAMP,
        origen TEXT, tabla TEXT, filas INTEGER)""")
    for tabla, csv in TABLAS.items():
        n = con.execute(f"SELECT COUNT(*) FROM {tabla}").fetchone()[0]
        con.execute("INSERT INTO sync_log (origen, tabla, filas) VALUES (?,?,?)",
                    ('carga_inicial_ML', tabla, n))
    con.commit()
    con.close()
    print(f"\nBD común lista: {db_path}")


def leer_tabla(tabla: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """Lee una tabla de la BD común (interfaz usada por el dashboard)."""
    con = sqlite3.connect(db_path)
    try:
        return pd.read_sql(f"SELECT * FROM {tabla}", con)
    finally:
        con.close()


def verificar(db_path: str = DB_PATH) -> None:
    con = sqlite3.connect(db_path)
    print(f"Esquema de {os.path.basename(db_path)}:")
    for (nombre,) in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        n = con.execute(f"SELECT COUNT(*) FROM {nombre}").fetchone()[0]
        cols = [c[1] for c in con.execute(f"PRAGMA table_info({nombre})")]
        print(f"  · {nombre} ({n} filas): {', '.join(cols[:6])}"
              + ("…" if len(cols) > 6 else ""))
    print("\nÚltimas sincronizaciones:")
    for fila in con.execute(
            "SELECT fecha, origen, tabla, filas FROM sync_log ORDER BY id DESC LIMIT 5"):
        print("  ", fila)
    con.close()


if __name__ == '__main__':
    if '--verificar' in sys.argv:
        verificar()
    else:
        crear_db()
