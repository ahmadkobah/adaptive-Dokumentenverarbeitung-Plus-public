"""
Configuration Settings — Plus Tier AI Service
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Server ────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "9001"))
API_TITLE = "Adaptive Document Processing — Plus Tier"
API_VERSION = "1.0.0"

# ── Database ──────────────────────────────────────────────
MONGO_URL = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "aura")

# ── Google Gemini (Plus = gemini-2.5-flash, Pro = gemini-2.5-pro) ───
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "32768"))

# ── Processing ────────────────────────────────────────────
IMAGE_DPI = int(os.getenv("IMAGE_DPI", "200"))
MAX_PDF_SIZE_MB = int(os.getenv("MAX_PDF_SIZE_MB", "500"))
MAX_PAGES = int(os.getenv("MAX_PAGES_PER_DOCUMENT", "100"))
POPPLER_PATH = os.getenv("POPPLER_PATH") or None

# ── Data Directories ──────────────────────────────────────
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
TEMP_DIR = DATA_DIR / "temp"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
