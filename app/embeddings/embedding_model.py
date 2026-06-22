"""Embedding generation layer.

This module loads a Sentence Transformers model once and converts security
text into dense embeddings for documents and queries.
"""

import logging
from typing import List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Generates vector embeddings for text using a Sentence Transformers model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Initialize and load the embedding model.

        Args:
            model_name: Name or path of the Sentence Transformers model to load.
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the Sentence Transformers model once."""
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as exc:
            logger.exception("Failed to load embedding model: %s", self.model_name)
            raise RuntimeError(
                f"Unable to load embedding model: {self.model_name}"
            ) from exc

        logger.info("Embedding model loaded: %s", self.model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents.

        Args:
            texts: List of text chunks to embed.

        Returns:
            A list of embedding vectors, one per input text.

        Raises:
            ValueError: If the input list is empty or contains empty text.
            RuntimeError: If the model is not loaded.
        """
        if not texts:
            logger.error("embed_documents called with empty text list")
            raise ValueError("Text list for embedding cannot be empty.")

        if self.model is None:
            logger.error("Embedding model is not loaded before embedding documents")
            raise RuntimeError("Embedding model is not loaded.")

        for text in texts:
            if not isinstance(text, str) or not text.strip():
                logger.error("Empty document text encountered")
                raise ValueError("Document text cannot be empty.")

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
        except Exception as exc:
            logger.exception("Failed to embed documents")
            raise RuntimeError("Document embedding failed.") from exc

        results = [vector.tolist() for vector in embeddings]
        logger.info("Number of documents embedded: %d", len(results))
        return results

    def embed_query(self, query: str) -> List[float]:
        """Generate an embedding for a single query string.

        Args:
            query: The query text to embed.

        Returns:
            The embedding vector for the query.

        Raises:
            ValueError: If the query is empty.
            RuntimeError: If the model is not loaded.
        """
        if not isinstance(query, str) or not query.strip():
            logger.error("embed_query called with empty query")
            raise ValueError("Query text cannot be empty.")

        if self.model is None:
            logger.error("Embedding model is not loaded before embedding query")
            raise RuntimeError("Embedding model is not loaded.")

        try:
            embedding = self.model.encode(query, convert_to_numpy=True)
        except Exception as exc:
            logger.exception("Failed to embed query")
            raise RuntimeError("Query embedding failed.") from exc

        result = embedding.tolist()
        logger.info("Query embedded")
        return result

    def get_dimension(self) -> int:
        """Return the dimensionality of the model embeddings."""
        if self.model is None:
            logger.error("get_dimension called before embedding model load")
            raise RuntimeError("Embedding model is not loaded.")

        return self.model.get_sentence_embedding_dimension()
