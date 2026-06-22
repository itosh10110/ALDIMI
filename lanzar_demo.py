"""
ALDIMI Predict — Lanzador de demo pública
=========================================
Ejecuta este script UNA VEZ para compartir el dashboard con tus compañeros.

Requisitos:
  pip install pyngrok

Uso:
  .venv\\Scripts\\python lanzar_demo.py
"""

import subprocess, sys, time, os, webbrowser

# ── 1. Verificar / instalar pyngrok ──────────────────────────────────────────
try:
    from pyngrok import ngrok, conf
except ImportError:
    print("📦 Instalando pyngrok...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok", "--quiet"])
    from pyngrok import ngrok, conf

# ── 2. Auth token de ngrok ────────────────────────────────────────────────────
TOKEN_FILE = os.path.join(os.path.dirname(__file__), ".ngrok_token")

print("\n" + "="*55)
print("  🏥  ALDIMI Predict — Demo pública con ngrok")
print("="*55)

if os.path.exists(TOKEN_FILE):
    with open(TOKEN_FILE) as f:
        token = f.read().strip()
    print(f"✅ Token ngrok encontrado en .ngrok_token")
else:
    print("""
ℹ️  Necesitas un token gratuito de ngrok (solo la primera vez):
   1. Ve a  https://dashboard.ngrok.com/signup  (es gratis)
   2. Copia tu authtoken desde  https://dashboard.ngrok.com/get-started/your-authtoken
   3. Pégalo aquí abajo ↓
""")
    token = input("   Pega tu ngrok authtoken: ").strip()
    if not token:
        print("❌ Token vacío. Saliendo.")
        sys.exit(1)
    with open(TOKEN_FILE, "w") as f:
        f.write(token)
    print(f"   (Token guardado en .ngrok_token para la próxima vez)")

conf.get_default().auth_token = token

# ── 3. Arrancar Streamlit en segundo plano ────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
dashboard_path = os.path.join(BASE, "dashboard.py")
venv_streamlit  = os.path.join(BASE, ".venv", "Scripts", "streamlit.exe")

# Usar streamlit del venv si existe, si no el del PATH
if os.path.exists(venv_streamlit):
    cmd = [venv_streamlit, "run", dashboard_path,
           "--server.port", "8501",
           "--server.headless", "true"]
else:
    cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path,
           "--server.port", "8501",
           "--server.headless", "true"]

print("\n🚀 Arrancando Streamlit en el puerto 8501...")
proc = subprocess.Popen(
    cmd,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    cwd=BASE,
)
time.sleep(4)  # darle tiempo a que levante

# ── 4. Abrir túnel ngrok ──────────────────────────────────────────────────────
print("🌐 Creando túnel público con ngrok...")
try:
    tunnel = ngrok.connect(8501, "http")
    url    = tunnel.public_url
    # ngrok a veces devuelve http, forzamos https
    url_https = url.replace("http://", "https://")
except Exception as e:
    print(f"❌ Error al conectar ngrok: {e}")
    proc.terminate()
    sys.exit(1)

# ── 5. Mostrar resultado ──────────────────────────────────────────────────────
print(f"""
{"="*55}
  ✅  Dashboard DISPONIBLE PÚBLICAMENTE

  🔗  {url_https}

  Comparte este link con tus compañeros.
  Funciona mientras esta ventana esté abierta.
{"="*55}

  Abriendo en tu navegador...
  (Presiona  Ctrl+C  para detener la demo)
""")

webbrowser.open(url_https)

# ── 6. Esperar hasta Ctrl+C ───────────────────────────────────────────────────
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n\n🛑 Demo detenida. El link ya no es accesible.")
    ngrok.disconnect(tunnel.public_url)
    proc.terminate()
    print("   Hasta la próxima. 👋\n")
