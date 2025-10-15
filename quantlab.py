# quantlab.py
from core.cli import run

# --- .env loader (preferred) ---
try:
    from pathlib import Path
    from dotenv import load_dotenv
    # project root is where quantlab.py lives
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path, override=False)
except Exception:
    pass
# --------------------------------

if __name__ == "__main__":
    run()
