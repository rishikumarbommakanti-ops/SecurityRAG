"""LOLBAS connector.

Fetches content from the official LOLBAS project site.
"""

import logging
from datetime import datetime
from typing import List

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


OFFICIAL_URLS = [
    "https://lolbas-project.github.io/",
]


def fetch_documents() -> List[dict]:
    docs = []
    for url in OFFICIAL_URLS:
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.string.strip() if soup.title else url
            paragraphs = soup.find_all("p")
            text = "\n\n".join(p.get_text(strip=True) for p in paragraphs[:50])

            if not text.strip():
                logger.warning("No extractable text from LOLBAS URL: %s", url)
                continue

            docs.append(
                {
                    "content": text,
                    "metadata": {
                        "source": url,
                        "category": "lolbas",
                        "title": title,
                        "retrieved_at": datetime.utcnow().isoformat() + "Z",
                    },
                }
            )
        except Exception as exc:
            logger.exception("Failed to fetch LOLBAS URL %s: %s", url, exc)
    return docs
