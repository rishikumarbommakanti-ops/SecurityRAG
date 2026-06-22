"""Tests for the knowledge manager."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.knowledge.knowledge_cache import (
    add_to_cache,
    fingerprint_document,
    is_duplicate,
    load_cache,
    save_cache,
)
from app.knowledge.knowledge_manager import KnowledgeManager


@pytest.fixture(autouse=True)
def clean_cache():
    """Clean knowledge cache before and after each test."""
    # Clean before test
    save_cache({})
    yield
    # Clean after test
    save_cache({})


@pytest.fixture
def mock_embedding_model():
    """Create mock embedding model."""
    model = MagicMock()
    model.embed_documents.return_value = [[0.1] * 384 for _ in range(5)]
    return model


@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    store = MagicMock()
    store.document_count.return_value = 100
    store.add_documents.return_value = None
    return store


@pytest.fixture
def knowledge_manager(mock_embedding_model, mock_vector_store):
    """Create knowledge manager with mocked dependencies."""
    return KnowledgeManager(
        embedding_model=mock_embedding_model, vector_store=mock_vector_store
    )


def test_fingerprint_document():
    """Test document fingerprinting for deduplication."""
    fp1 = fingerprint_document("content", "source1", "title1")
    fp2 = fingerprint_document("content", "source1", "title1")
    fp3 = fingerprint_document("content", "source2", "title1")

    assert fp1 == fp2
    assert fp1 != fp3
    assert len(fp1) == 64


def test_is_duplicate():
    """Test duplicate detection."""
    content = "This is a security document"
    source = "MITRE"
    title = "T1234"

    # Initially should not be duplicate
    assert not is_duplicate(content, source, title)

    # After adding to cache, should be duplicate
    add_to_cache(content, source, title, {})
    assert is_duplicate(content, source, title)


def test_knowledge_manager_update_source_empty(knowledge_manager, mock_vector_store):
    """Test updating a source with no new documents."""
    with patch("app.knowledge.knowledge_manager.CONNECTORS", {"mitre": lambda: []}):
        result = knowledge_manager.update_source("mitre")

    assert result["source"] == "mitre"
    assert result["new_documents"] == 0
    mock_vector_store.add_documents.assert_not_called()


def test_knowledge_manager_update_source_with_documents(
    knowledge_manager, mock_embedding_model, mock_vector_store
):
    """Test updating a source with new documents."""
    documents = [
        {
            "content": "Technique T1234 description",
            "metadata": {
                "source": "https://attack.mitre.org",
                "title": "T1234: Process Execution",
                "category": "technique",
            },
        }
    ]

    with patch("app.knowledge.knowledge_manager.CONNECTORS", {"mitre": lambda: documents}):
        result = knowledge_manager.update_source("mitre")

    assert result["source"] == "mitre"
    assert result["new_documents"] == 1
    mock_vector_store.add_documents.assert_called_once()
    mock_embedding_model.embed_documents.assert_called_once()


def test_knowledge_manager_update_all(knowledge_manager):
    """Test updating all sources."""
    mock_documents = [
        {
            "content": "Sample content",
            "metadata": {
                "source": "https://example.com",
                "title": "Sample",
            },
        }
    ]

    with patch("app.connectors.mitre_connector.fetch_documents", return_value=mock_documents), patch(
        "app.connectors.sigma_connector.fetch_documents", return_value=mock_documents
    ), patch(
        "app.connectors.cisa_connector.fetch_documents", return_value=mock_documents
    ), patch(
        "app.connectors.microsoft_connector.fetch_documents", return_value=mock_documents
    ), patch(
        "app.connectors.lolbas_connector.fetch_documents", return_value=mock_documents
    ):
        results = knowledge_manager.update_all()

    assert len(results) >= 5
    assert all(result["source"] in ["mitre", "sigma", "cisa", "microsoft", "lolbas"] for result in results)


def test_knowledge_manager_duplicate_prevention(
    knowledge_manager, mock_embedding_model
):
    """Test that duplicate documents are skipped."""
    content = "Duplicate content for testing"
    source = "https://example.com"
    title = "Example Document"

    # Pre-populate cache with the exact content
    add_to_cache(content, source, title, {})

    # Verify it's in cache
    assert is_duplicate(content, source, title)

    documents = [
        {
            "content": content,
            "metadata": {"source": source, "title": title},
        }
    ]

    with patch("app.knowledge.knowledge_manager.CONNECTORS", {"mitre": lambda: documents}):
        result = knowledge_manager.update_source("mitre")

    assert result["new_documents"] == 0
    mock_embedding_model.embed_documents.assert_not_called()


def test_knowledge_manager_document_statistics(knowledge_manager, mock_vector_store):
    """Test document statistics reporting."""
    # Add documents to cache directly
    add_to_cache("Content1", "MITRE", "T1234", {})
    add_to_cache("Content2", "MITRE", "T5678", {})
    add_to_cache("Content3", "Sigma", "Rule1", {})

    stats = knowledge_manager.document_statistics()

    assert stats["total_documents"] == 100  # from mock_vector_store
    assert stats["per_source"]["MITRE"] == 2
    assert stats["per_source"]["Sigma"] == 1


def test_knowledge_manager_update_unknown_source(knowledge_manager):
    """Test error handling for unknown source."""
    with pytest.raises(ValueError):
        knowledge_manager.update_source("unknown_source")


def test_knowledge_manager_handles_fetch_errors(knowledge_manager, mock_vector_store):
    """Test error handling when fetch fails."""
    def raise_error():
        raise Exception("Network error")
    
    with patch("app.knowledge.knowledge_manager.CONNECTORS", {"mitre": raise_error}):
        with pytest.raises(Exception, match="Network error"):
            knowledge_manager.update_source("mitre")


def test_knowledge_manager_handles_embedding_errors(
    knowledge_manager, mock_embedding_model, mock_vector_store
):
    """Test error handling when embedding fails."""
    documents = [
        {
            "content": "Test content",
            "metadata": {"source": "https://example.com", "title": "Test"},
        }
    ]

    mock_embedding_model.embed_documents.side_effect = Exception(
        "Embedding model error"
    )

    with patch("app.knowledge.knowledge_manager.CONNECTORS", {"mitre": lambda: documents}):
        with pytest.raises(Exception, match="Embedding model error"):
            knowledge_manager.update_source("mitre")
