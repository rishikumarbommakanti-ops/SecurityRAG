"""Vector database implementation using ChromaDB.

This module manages a ChromaDB collection for storing and querying embedded
security documents.
"""

import logging
import uuid
from pathlib import Path
from typing import List

from chromadb import Client
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class ChromaStore:
    """Manages a ChromaDB collection for storing and querying document embeddings."""

    def __init__(self, persist_directory: Path, collection_name: str) -> None:
        """Initialize the vector store configuration.

        Args:
            persist_directory: Directory where ChromaDB persists its data.
            collection_name: Name of the ChromaDB collection to use.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None

    def initialize_collection(self) -> None:
        """Create or open the persistent ChromaDB collection."""
        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            settings = Settings(
                persist_directory=str(self.persist_directory),
                is_persistent=True,
            )
            self.client = Client(settings)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
        except Exception as exc:
            logger.exception("Failed to initialize ChromaDB collection: %s", self.collection_name)
            raise RuntimeError(
                f"Unable to initialize ChromaDB collection: {self.collection_name}"
            ) from exc

        logger.info("Collection initialized: %s", self.collection_name)

    def _ensure_collection(self) -> None:
        if self.collection is None:
            self.initialize_collection()

    def add_documents(self, documents: List[dict], embeddings: List[List[float]]) -> None:
        """Add documents and embeddings to the ChromaDB collection.

        Args:
            documents: List of document dictionaries with content and metadata.
            embeddings: List of embedding vectors corresponding to the documents.
        """
        if not documents:
            logger.error("add_documents called with empty documents list")
            raise ValueError("Documents list cannot be empty.")

        if not embeddings:
            logger.error("add_documents called with empty embeddings list")
            raise ValueError("Embeddings list cannot be empty.")

        if len(documents) != len(embeddings):
            logger.error(
                "Mismatch between documents (%d) and embeddings (%d)",
                len(documents),
                len(embeddings),
            )
            raise ValueError("Documents and embeddings must have the same length.")

        contents = []
        metadatas = []
        ids = []

        for index, document in enumerate(documents):
            if not isinstance(document, dict):
                raise ValueError("Each document must be a dictionary.")

            content = document.get("content")
            metadata = document.get("metadata")

            if not isinstance(content, str) or not content.strip():
                logger.error("Invalid document content at index %d", index)
                raise ValueError("Each document must include non-empty content.")

            if not isinstance(metadata, dict):
                logger.error("Invalid document metadata at index %d", index)
                raise ValueError("Each document must include metadata as a dictionary.")

            contents.append(content)
            metadatas.append(metadata)
            ids.append(str(uuid.uuid4()))

        for embedding_index, embedding in enumerate(embeddings):
            if not isinstance(embedding, (list, tuple)) or not embedding:
                logger.error("Invalid embedding at index %d", embedding_index)
                raise ValueError("Each embedding must be a non-empty list of numbers.")

            if not all(isinstance(value, (int, float)) for value in embedding):
                logger.error("Non-numeric embedding values at index %d", embedding_index)
                raise ValueError("Embedding values must be numeric.")

        self._ensure_collection()

        try:
            self.collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        except Exception as exc:
            logger.exception("Failed to add documents to ChromaDB collection: %s", self.collection_name)
            raise RuntimeError("Failed to add documents to the vector store.") from exc

        logger.info("Documents added: %d to collection %s", len(documents), self.collection_name)

    def similarity_search(self, query_embedding: List[float], top_k: int) -> List[dict]:
        """Search the collection for documents similar to the query embedding."""
        if not isinstance(query_embedding, (list, tuple)) or not query_embedding:
            logger.error("similarity_search called with invalid query_embedding")
            raise ValueError("Query embedding must be a non-empty list of numbers.")

        if not all(isinstance(value, (int, float)) for value in query_embedding):
            logger.error("similarity_search received non-numeric query_embedding")
            raise ValueError("Query embedding values must be numeric.")

        if not isinstance(top_k, int) or top_k <= 0:
            logger.error("similarity_search called with invalid top_k: %s", top_k)
            raise ValueError("top_k must be a positive integer.")

        self._ensure_collection()

        if self.document_count() == 0:
            logger.warning("Similarity search performed on empty collection: %s", self.collection_name)
            return []

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.exception("ChromaDB search failed for collection: %s", self.collection_name)
            raise RuntimeError("Vector store similarity search failed.") from exc

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        response = []
        for content, metadata, distance in zip(documents, metadatas, distances):
            if content is None:
                continue

            score = 1.0 / (1.0 + float(distance))
            response.append(
                {
                    "content": content,
                    "metadata": metadata,
                    "score": score,
                }
            )

        logger.info(
            "Similarity search performed: %d results returned from collection %s",
            len(response),
            self.collection_name,
        )

        return response

    def document_count(self) -> int:
        """Return the number of documents currently stored in the collection."""
        self._ensure_collection()

        try:
            return self.collection.count()
        except Exception as exc:
            logger.exception("Failed to count documents in collection: %s", self.collection_name)
            raise RuntimeError("Unable to retrieve document count from the vector store.") from exc

    def clear_collection(self) -> None:
        """Remove all documents from the collection."""
        self._ensure_collection()

        try:
            result = self.collection.peek(limit=1000)
            ids = result.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
        except Exception as exc:
            logger.exception("Failed to clear collection: %s", self.collection_name)
            raise RuntimeError("Unable to clear the vector store collection.") from exc

        logger.info("Collection cleared: %s", self.collection_name)
