import pytest

from app.embeddings.embedding_model import EmbeddingModel


def test_embedding_model_initialization() -> None:
    model = EmbeddingModel()
    assert model.model is not None
    assert model.get_dimension() > 0


def test_embed_documents_returns_vectors() -> None:
    model = EmbeddingModel()
    texts = ["This is a test document.", "Another security text chunk."]
    embeddings = model.embed_documents(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(isinstance(vector, list) for vector in embeddings)
    assert all(len(vector) == model.get_dimension() for vector in embeddings)


def test_embed_query_returns_vector() -> None:
    model = EmbeddingModel()
    embedding = model.embed_query("Explain T1059.")

    assert isinstance(embedding, list)
    assert len(embedding) == model.get_dimension()


def test_embed_documents_rejects_empty_text() -> None:
    model = EmbeddingModel()

    with pytest.raises(ValueError):
        model.embed_documents([""])


def test_embed_query_rejects_empty_query() -> None:
    model = EmbeddingModel()

    with pytest.raises(ValueError):
        model.embed_query("")
