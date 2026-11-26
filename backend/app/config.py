"""Paths and constants. Everything lives under the project folder so the app stays portable."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PROJECTS_DIR = DATA_DIR / "projects"
BRAND_DIR = DATA_DIR / "brand"
KITS_DIR = DATA_DIR / "kits"
LOG_DIR = DATA_DIR / "logs"
BROLL_CACHE = DATA_DIR / "broll_cache"
DB_PATH = DATA_DIR / "clipforge.db"
FRONTEND_DIR = BASE_DIR / "frontend"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
BRAND_DIR.mkdir(parents=True, exist_ok=True)
KITS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    """Errors and tracebacks go to data/logs/clipforge.log (rotated), so a problem can be sent for diagnosis."""
    import logging
    from logging.handlers import RotatingFileHandler
    log = logging.getLogger("clipforge")
    if log.handlers:
        return
    h = RotatingFileHandler(LOG_DIR / "clipforge.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)
