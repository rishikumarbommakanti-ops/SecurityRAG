"""Simple persistent cache to prevent duplicate ingestion.

Stores a small JSON index of seen document fingerprints and metadata.
"""

import json
import logging
from hashlib import sha256
from pathlib import Path
from typing import Dict

from app import config

logger = logging.getLogger(__name__)


CACHE_PATH = Path(config.BASE_DIR) / "data" / "knowledge_cache.json"


def _ensure_cache_file() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE_PATH.exists():
        CACHE_PATH.write_text(json.dumps({}))


def load_cache() -> Dict[str, dict]:
    _ensure_cache_file()
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to load knowledge cache, resetting")
        CACHE_PATH.write_text(json.dumps({}))
        return {}


def save_cache(cache: Dict[str, dict]) -> None:
    _ensure_cache_file()
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def fingerprint_document(content: str, source: str, title: str) -> str:
    key = (source or "") + "|" + (title or "") + "|" + (content or "")[:4096]
    return sha256(key.encode("utf-8")).hexdigest()


def is_duplicate(content: str, source: str, title: str) -> bool:
    cache = load_cache()
    fp = fingerprint_document(content, source, title)
    return fp in cache


def add_to_cache(content: str, source: str, title: str, metadata: dict) -> None:
    cache = load_cache()
    fp = fingerprint_document(content, source, title)
    cache[fp] = {
        "source": source,
        "title": title,
        "metadata": metadata,
    }
    save_cache(cache)
