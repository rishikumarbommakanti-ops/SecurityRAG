import logging
from pathlib import Path

import pytest

from app.ingestion.document_loader import DocumentLoader


LOGGER = logging.getLogger(__name__)


def test_load_pdf_missing_file(tmp_path: Path) -> None:
    loader = DocumentLoader(source_dir=tmp_path)
    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        loader.load_pdf(missing_pdf)


def test_load_pdf_empty_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")

    loader = DocumentLoader(source_dir=tmp_path)
    with pytest.raises(ValueError):
        loader.load_pdf(pdf_path)


def test_split_documents_returns_chunks(tmp_path: Path) -> None:
    loader = DocumentLoader(source_dir=tmp_path)
    documents = [
        {
            "content": "".join(["word " for _ in range(1200)]),
            "metadata": {"source": "test.pdf", "page": 1},
        }
    ]

    chunked = loader.split_documents(documents)

    assert chunked, "Expected chunked documents"
    assert all("content" in item and "metadata" in item for item in chunked)
    assert chunked[0]["metadata"]["source"] == "test.pdf"
    assert chunked[0]["metadata"]["page"] == 1


def test_load_documents_with_pdf(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n")

    loader = DocumentLoader(source_dir=tmp_path)

    class DummyDocument:
        def __init__(self, page_content: str):
            self.page_content = page_content
            self.metadata = {}

    def dummy_load(self):
        return [DummyDocument("This is a test page.")]

    monkeypatch.setattr(
        "app.ingestion.document_loader.PyPDFLoader.load",
        dummy_load,
    )

    chunked_documents = loader.load_documents()

    assert len(chunked_documents) == 1
    assert chunked_documents[0]["metadata"]["source"] == "sample.pdf"
    assert chunked_documents[0]["metadata"]["page"] == 1
    assert "This is a test page." in chunked_documents[0]["content"]
