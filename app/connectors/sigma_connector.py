"""Sigma rules connector.

Fetches content from the official SigmaHQ repository (README) as an
authoritative source of Sigma information.
"""

import logging
from datetime import datetime
from typing import List

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


OFFICIAL_URLS = [
    "https://raw.githubusercontent.com/SigmaHQ/sigma/master/README.md",
]


def fetch_documents() -> List[dict]:
    docs = []
    for url in OFFICIAL_URLS:
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            text = resp.text
            title = "Sigma Rules Repository"
            if not text.strip():
                logger.warning("No extractable text from Sigma URL: %s", url)
                continue

            docs.append(
                {
                    "content": text,
                    "metadata": {
                        "source": url,
                        "category": "sigma",
                        "title": title,
                        "retrieved_at": datetime.utcnow().isoformat() + "Z",
                    },
                }
            )
        except Exception as exc:
            logger.exception("Failed to fetch Sigma URL %s: %s", url, exc)
    return docs
