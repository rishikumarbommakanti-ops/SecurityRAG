"""Knowledge manager orchestrates connectors, embedding, and ingestion.
"""

import logging
from datetime import datetime
from typing import Callable, Dict, List

from app.connectors import mitre_connector, sigma_connector, cisa_connector, microsoft_connector, lolbas_connector
from app.embeddings.embedding_model import EmbeddingModel
from app.knowledge.knowledge_cache import add_to_cache, is_duplicate, load_cache
from app.vectorstore.chroma_store import ChromaStore
from app import config

logger = logging.getLogger(__name__)


CONNECTORS: Dict[str, Callable[[], List[dict]]] = {
    "mitre": mitre_connector.fetch_documents,
    "sigma": sigma_connector.fetch_documents,
    "cisa": cisa_connector.fetch_documents,
    "microsoft": microsoft_connector.fetch_documents,
    "lolbas": lolbas_connector.fetch_documents,
}


class KnowledgeManager:
    def __init__(self, embedding_model: EmbeddingModel, vector_store: ChromaStore) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def update_source(self, source_name: str) -> dict:
        """Fetch, deduplicate, embed, and store documents from a named source.

        Returns a summary dict with counts and timestamps.
        """
        if source_name not in CONNECTORS:
            raise ValueError(f"Unknown source: {source_name}")

        fetcher = CONNECTORS[source_name]
        documents = fetcher()

        new_docs = []
        contents = []
        metadatas = []
        for doc in documents:
            content = doc.get("content", "").strip()
            metadata = doc.get("metadata", {})
            title = metadata.get("title")
            source = metadata.get("source")
            if not content:
                continue
            if is_duplicate(content, source, title):
                logger.info("Skipping duplicate document from %s", source)
                continue
            new_docs.append(doc)
            contents.append(content)
            metadatas.append(metadata)

        if not new_docs:
            return {"source": source_name, "new_documents": 0, "updated_at": datetime.utcnow().isoformat() + "Z"}

        # generate embeddings
        embeddings = self.embedding_model.embed_documents(contents)

        # add to vector store (Chroma)
        try:
            self.vector_store.add_documents(new_docs, embeddings)
        except Exception:
            logger.exception("Failed to add documents to vector store for source %s", source_name)
            raise

        # update cache
        for doc in new_docs:
            meta = doc.get("metadata", {})
            add_to_cache(doc.get("content", ""), meta.get("source"), meta.get("title"), meta)

        return {"source": source_name, "new_documents": len(new_docs), "updated_at": datetime.utcnow().isoformat() + "Z"}

    def update_all(self) -> List[dict]:
        results = []
        for name in CONNECTORS.keys():
            try:
                results.append(self.update_source(name))
            except Exception:
                logger.exception("Failed to update source %s", name)
        return results

    def document_statistics(self) -> dict:
        cache = load_cache()
        total_documents = self.vector_store.document_count()
        per_source = {}
        for entry in cache.values():
            src = entry.get("source") or "unknown"
            per_source[src] = per_source.get(src, 0) + 1

        return {"total_documents": total_documents, "per_source": per_source}
