"""Central configuration for SecurityRAG.

Loads secrets and environment variables and exposes project-wide settings.
"""

import os
from pathlib import Path
from typing import Any

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"


def _get_secret(key: str, default: Any = "") -> Any:
    if hasattr(st, "secrets"):
        try:
            return st.secrets.get(key, os.getenv(key, default))
        except FileNotFoundError:
            return os.getenv(key, default)
        except Exception:
            return os.getenv(key, default)
    return os.getenv(key, default)


OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
GOOGLE_API_KEY = _get_secret("GOOGLE_API_KEY")
LLM_MODEL_NAME = _get_secret(
    "GEMINI_MODEL", os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash-002")
)
GEMINI_TEMPERATURE = float(
    _get_secret("GEMINI_TEMPERATURE", os.getenv("GEMINI_TEMPERATURE", "0.1"))
)
ENABLE_GEMINI_FALLBACK = _get_secret(
    "ENABLE_GEMINI_FALLBACK", os.getenv("ENABLE_GEMINI_FALLBACK", "true")
).lower() in ("1", "true", "yes")
EMBEDDING_MODEL_NAME = _get_secret(
    "EMBEDDING_MODEL_NAME", os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
)
CHROMA_COLLECTION_NAME = _get_secret(
    "CHROMA_COLLECTION_NAME", os.getenv("CHROMA_COLLECTION_NAME", "security_knowledge")
)
RETRIEVER_TOP_K = int(_get_secret("RETRIEVER_TOP_K", os.getenv("RETRIEVER_TOP_K", "4")))
SECURITY_PROMPT_PATH = BASE_DIR / "app" / "prompts" / "security_prompt.txt"
ENABLE_LIVE_UPDATES = _get_secret("ENABLE_LIVE_UPDATES", os.getenv("ENABLE_LIVE_UPDATES", "false")).lower() in ("1", "true", "yes")
KNOWLEDGE_REFRESH_HOURS = int(_get_secret("KNOWLEDGE_REFRESH_HOURS", os.getenv("KNOWLEDGE_REFRESH_HOURS", "24")))
KNOWLEDGE_CACHE_PATH = BASE_DIR / "data" / "knowledge_cache.json"
DEMO_MODE = _get_secret("DEMO_MODE", os.getenv("DEMO_MODE", "false")).lower() in ("1", "true", "yes")

if DEMO_MODE:
    ENABLE_LIVE_UPDATES = False


def get_api_status() -> dict:
    return {
        "gemini_key_present": bool(GOOGLE_API_KEY),
        "gemini_model": LLM_MODEL_NAME,
        "gemini_fallback": ENABLE_GEMINI_FALLBACK,
        "demo_mode": DEMO_MODE,
        "live_updates_enabled": ENABLE_LIVE_UPDATES,
    }
