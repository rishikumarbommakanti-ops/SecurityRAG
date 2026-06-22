import shutil
from pathlib import Path

import pytest

from app.vectorstore.chroma_store import ChromaStore


def test_initialize_collection(tmp_path: Path) -> None:
    store = ChromaStore(persist_directory=tmp_path, collection_name="security_knowledge")
    store.initialize_collection()

    assert store.collection is not None
    assert store.document_count() == 0


def test_add_documents_and_document_count(tmp_path: Path) -> None:
    store = ChromaStore(persist_directory=tmp_path, collection_name="security_knowledge")
    store.initialize_collection()

    documents = [
        {"content": "Test document 1", "metadata": {"source": "doc1.pdf"}},
        {"content": "Test document 2", "metadata": {"source": "doc2.pdf"}},
    ]
    embeddings = [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]

    store.add_documents(documents, embeddings)

    assert store.document_count() == 2


def test_similarity_search_returns_results(tmp_path: Path) -> None:
    store = ChromaStore(persist_directory=tmp_path, collection_name="security_knowledge")
    store.initialize_collection()

    documents = [
        {"content": "Security alert about malware.", "metadata": {"source": "alert.pdf"}},
        {"content": "Network traffic investigation notes.", "metadata": {"source": "traffic.pdf"}},
    ]
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    store.add_documents(documents, embeddings)
    results = store.similarity_search([0.9, 0.1, 0.0], top_k=1)

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["content"] == "Security alert about malware."
    assert results[0]["metadata"]["source"] == "alert.pdf"
    assert 0.0 <= results[0]["score"] <= 1.0


def test_similarity_search_on_empty_collection_returns_empty(tmp_path: Path) -> None:
    store = ChromaStore(persist_directory=tmp_path, collection_name="security_knowledge")
    store.initialize_collection()

    results = store.similarity_search([0.1, 0.2, 0.3], top_k=3)

    assert results == []


def test_clear_collection(tmp_path: Path) -> None:
    store = ChromaStore(persist_directory=tmp_path, collection_name="security_knowledge")
    store.initialize_collection()

    documents = [{"content": "Sample doc", "metadata": {"source": "sample.pdf"}}]
    store.add_documents(documents, [[0.1, 0.2, 0.3]])

    assert store.document_count() == 1

    store.clear_collection()
    assert store.document_count() == 0


def test_add_documents_rejects_invalid_inputs(tmp_path: Path) -> None:
    store = ChromaStore(persist_directory=tmp_path, collection_name="security_knowledge")
    store.initialize_collection()

    with pytest.raises(ValueError):
        store.add_documents([], [[0.1]])

    with pytest.raises(ValueError):
        store.add_documents([{"content": "", "metadata": {"source": "doc.pdf"}}], [[0.1]])


def test_similarity_search_rejects_invalid_query_embedding(tmp_path: Path) -> None:
    store = ChromaStore(persist_directory=tmp_path, collection_name="security_knowledge")
    store.initialize_collection()

    with pytest.raises(ValueError):
        store.similarity_search([], top_k=1)

    with pytest.raises(ValueError):
        store.similarity_search(["invalid"], top_k=1)

    with pytest.raises(ValueError):
        store.similarity_search([0.1, 0.2], top_k=0)
