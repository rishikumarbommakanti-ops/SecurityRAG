from pathlib import Path

import pytest

from app.embeddings.embedding_model import EmbeddingModel
from app.retrieval.retriever import SecurityRetriever
from app.vectorstore.chroma_store import ChromaStore


class DummyEmbeddingModel:
    def __init__(self, dimension: int = 3):
        self.dimension = dimension

    def embed_query(self, query: str):
        return [float(len(query)), 0.0, 0.0]


class DummyVectorStore:
    def __init__(self):
        self.documents = []

    def initialize_collection(self):
        self.documents = []

    def document_count(self):
        return len(self.documents)

    def similarity_search(self, query_embedding, top_k):
        if not self.documents:
            return []
        return self.documents[:top_k]


def test_retrieve_returns_documents(tmp_path: Path) -> None:
    embedding_model = DummyEmbeddingModel()
    vector_store = DummyVectorStore()
    retriever = SecurityRetriever(embedding_model, vector_store, top_k=2)

    vector_store.documents = [
        {"content": "Document 1", "metadata": {"source": "doc1.pdf"}, "score": 0.9},
        {"content": "Document 2", "metadata": {"source": "doc2.pdf"}, "score": 0.8},
    ]

    results = retriever.retrieve("Test query")

    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0]["content"] == "Document 1"


def test_retrieve_with_scores_limited_by_top_k(tmp_path: Path) -> None:
    embedding_model = DummyEmbeddingModel()
    vector_store = DummyVectorStore()
    retriever = SecurityRetriever(embedding_model, vector_store, top_k=1)

    vector_store.documents = [
        {"content": "Document 1", "metadata": {"source": "doc1.pdf"}, "score": 0.9},
        {"content": "Document 2", "metadata": {"source": "doc2.pdf"}, "score": 0.8},
    ]

    results = retriever.retrieve_with_scores("Test query")

    assert len(results) == 1
    assert results[0]["score"] == 0.9


def test_retrieve_rejects_empty_query(tmp_path: Path) -> None:
    embedding_model = DummyEmbeddingModel()
    vector_store = DummyVectorStore()
    retriever = SecurityRetriever(embedding_model, vector_store, top_k=1)

    with pytest.raises(ValueError):
        retriever.retrieve("")


def test_format_context_returns_structured_string(tmp_path: Path) -> None:
    embedding_model = DummyEmbeddingModel()
    vector_store = DummyVectorStore()
    retriever = SecurityRetriever(embedding_model, vector_store, top_k=1)

    documents = [
        {"content": "Test content.", "metadata": {"source": "doc1.pdf"}},
    ]

    formatted = retriever.format_context(documents)

    assert "Source: doc1.pdf" in formatted
    assert "Content:" in formatted
    assert "Test content." in formatted


def test_retrieve_returns_empty_when_no_results(tmp_path: Path) -> None:
    embedding_model = DummyEmbeddingModel()
    vector_store = DummyVectorStore()
    retriever = SecurityRetriever(embedding_model, vector_store, top_k=3)

    vector_store.documents = []
    results = retriever.retrieve("Test query")

    assert results == []
