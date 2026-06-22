"""MITRE ATT&CK connector.

Fetches content from the official MITRE ATT&CK website and extracts
relevant text for ingestion.
"""

import logging
from datetime import datetime
from typing import List

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


OFFICIAL_URLS = [
    "https://attack.mitre.org/",
]


def fetch_documents() -> List[dict]:
    docs = []
    for url in OFFICIAL_URLS:
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.string.strip() if soup.title else url
            # Prefer article/main content when available, otherwise join paragraphs
            main = soup.find("main") or soup.find("article")
            if main:
                text = "\n\n".join(p.get_text(strip=True) for p in main.find_all("p"))
            else:
                paragraphs = soup.find_all("p")
                text = "\n\n".join(p.get_text(strip=True) for p in paragraphs[:20])

            if not text.strip():
                logger.warning("No extractable text from MITRE URL: %s", url)
                continue

            docs.append(
                {
                    "content": text,
                    "metadata": {
                        "source": url,
                        "category": "mitre",
                        "title": title,
                        "retrieved_at": datetime.utcnow().isoformat() + "Z",
                    },
                }
            )
        except Exception as exc:
            logger.exception("Failed to fetch MITRE URL %s: %s", url, exc)
    return docs
