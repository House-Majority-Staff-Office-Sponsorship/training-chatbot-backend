"""
Centralized configuration loaded from environment variables.
Mirrors the Node.js backend's env var names exactly.
"""

import os
import json

from dotenv import load_dotenv

load_dotenv()

# ── Google Cloud / Vertex AI ─────────────────────────────────────────────
GCP_PROJECT = os.getenv("GCP_PROJECT", "")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-west1")

# Tell ADK to use Vertex AI
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
if GCP_PROJECT:
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", GCP_PROJECT)
if GCP_LOCATION:
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", GCP_LOCATION)

# ── Model configuration ─────────────────────────────────────────────────
GEN_FAST_MODEL = os.getenv("GEN_FAST_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
GEN_REPORT_MODEL = os.getenv("GEN_REPORT_MODEL", "gemini-2.5-pro")
GEN_PRO_MODEL = os.getenv("GEN_PRO_MODEL", GEN_REPORT_MODEL)

# ── RAG corpus ──────────────────────────────────────────────────────────
RAG_CORPUS = os.getenv("RAG_CORPUS", "")

# ── API Key ─────────────────────────────────────────────────────────────
API_KEY = os.getenv("API_KEY", "")

# ── CORS ────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

# ── Service Account Key (for non-GCP environments like Coolify) ─────────
_creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "")
if _creds_json and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    # Write service account key to a temp file and set the env var
    import tempfile
    _creds_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    _creds_file.write(_creds_json)
    _creds_file.close()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _creds_file.name


def require_config():
    """Raise HTTPException if required config is missing."""
    from fastapi import HTTPException
    if not GCP_PROJECT:
        raise HTTPException(status_code=500, detail="Server misconfiguration: GCP_PROJECT is not set.")
    if not RAG_CORPUS:
        raise HTTPException(status_code=500, detail="Server misconfiguration: RAG_CORPUS is not set.")
