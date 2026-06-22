"""Retrieval pipeline implementation.

This module orchestrates the retrieval flow: embed a user query, search the
ChromaDB vector store, and return structured results ready for downstream LLM
context construction.
"""

import logging
import time
from typing import List

logger = logging.getLogger(__name__)


class SecurityRetriever:
    """Coordinates query embedding and vector store search for the RAG pipeline."""

    def __init__(self, embedding_model, vector_store, top_k: int) -> None:
        """Initialize the retriever with its dependencies.

        Args:
            embedding_model: An embedding model instance used to embed queries.
            vector_store: A vector store instance used to search for documents.
            top_k: Number of top documents to retrieve per query.
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k

    def initialize(self) -> None:
        """Initialize the retrieval pipeline dependencies."""
        logger.info("Initializing retrieval pipeline")

        try:
            if hasattr(self.vector_store, "initialize_collection"):
                self.vector_store.initialize_collection()
        except Exception as exc:
            logger.exception("Failed to initialize retrieval pipeline")
            raise RuntimeError("Failed to initialize the retrieval pipeline.") from exc

        logger.info("Retrieval pipeline initialized")

    def retrieve(self, query: str) -> List[dict]:
        """Retrieve relevant documents for a query.

        Args:
            query: The natural language question asked by the user.

        Returns:
            A list of retrieved documents with content, metadata, and score.
        """
        return self.retrieve_with_scores(query)

    def retrieve_with_scores(self, query: str) -> List[dict]:
        """Retrieve relevant documents for a query and return scores."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Empty query received for retrieval")
            raise ValueError("Query cannot be empty.")

        logger.info("Query received: %s", query)
        start_time = time.perf_counter()

        try:
            query_embedding = self.embedding_model.embed_query(query)
        except Exception as exc:
            logger.exception("Failed to generate query embedding")
            raise RuntimeError("Query embedding failed.") from exc

        logger.info("Query embedded")

        try:
            if hasattr(self.vector_store, "document_count") and self.vector_store.document_count() == 0:
                logger.warning("Retrieval requested on empty collection")
                return []
        except Exception as exc:
            logger.exception("Failed to check collection state")
            raise RuntimeError("Unable to verify vector store collection state.") from exc

        try:
            documents = self.vector_store.similarity_search(query_embedding, self.top_k)
        except Exception as exc:
            logger.exception("Similarity search failed")
            raise RuntimeError("Retrieval failed during similarity search.") from exc

        elapsed = time.perf_counter() - start_time
        logger.info("Documents retrieved: %d in %.3fs", len(documents), elapsed)

        return documents or []

    def format_context(self, documents: List[dict]) -> str:
        """Format retrieved documents into a structured context string."""
        if not documents:
            logger.warning("format_context called with no documents")
            return ""

        formatted_entries = []
        for document in documents:
            metadata = document.get("metadata") if isinstance(document, dict) else None
            source = None
            if isinstance(metadata, dict):
                source = metadata.get("source") or metadata.get("source_name")
            source_text = source if isinstance(source, str) and source.strip() else "Unknown source"

            content = document.get("content", "") if isinstance(document, dict) else ""
            formatted_entries.append(
                f"Source: {source_text}\n\nContent:\n{content.strip()}"
            )

        result = "\n\n---\n\n".join(formatted_entries)
        logger.info("Context formatted for %d documents", len(formatted_entries))
        return result
