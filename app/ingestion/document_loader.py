"""Document ingestion implementation.

This module loads PDF files from the data directory, extracts text using
PyPDFLoader, splits long text into chunks, and returns normalized document
objects with metadata.
"""

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.schema import Document
from langchain_text_splitters import CharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads and normalizes security documents from the data directory."""

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    def __init__(self, source_dir: Path) -> None:
        """Initialize the loader with the directory containing raw documents.

        Args:
            source_dir: Path to the directory holding source documents.
        """
        self.source_dir = source_dir

    def load_pdf(self, file_path: Path) -> List[dict]:
        """Load a PDF file and return page-level text documents.

        Args:
            file_path: Path to the PDF file.

        Returns:
            A list of dictionaries with page content and metadata.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the PDF cannot be parsed.
        """
        if not file_path.exists():
            logger.error("PDF missing: %s", file_path)
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
        except Exception as exc:
            logger.exception("Failed to load PDF %s", file_path)
            raise ValueError(f"Unable to read PDF file: {file_path}") from exc

        logger.info("PDF loaded: %s", file_path.name)
        logger.info("Pages extracted: %d", len(documents))

        page_documents: List[dict] = []
        for page_number, document in enumerate(documents, start=1):
            content = document.page_content.strip()
            if not content:
                logger.warning(
                    "Empty page skipped for %s page %d", file_path.name, page_number
                )
                continue

            page_documents.append(
                {
                    "content": content,
                    "metadata": {
                        "source": file_path.name,
                        "page": page_number,
                    },
                }
            )

        if not page_documents:
            logger.warning("No text content extracted from PDF: %s", file_path.name)

        return page_documents

    def split_documents(self, documents: List[dict]) -> List[dict]:
        """Split documents into smaller chunks for embedding.

        Args:
            documents: A list of page-level documents.

        Returns:
            A list of chunked document dictionaries.
        """
        if not documents:
            return []

        splitter = CharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
        )

        source_documents = [
            Document(page_content=item["content"], metadata=item["metadata"])
            for item in documents
        ]

        chunked_documents = splitter.split_documents(source_documents)
        chunked_results = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in chunked_documents
        ]

        logger.info("Chunks created: %d", len(chunked_results))
        return chunked_results

    def load_documents(self) -> List[dict]:
        """Load and split all PDF documents from the source directory.

        Returns:
            A list of chunked document dictionaries ready for ingestion.
        """
        if not self.source_dir.exists():
            logger.error("Source directory missing: %s", self.source_dir)
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")

        pdf_files = sorted(self.source_dir.glob("*.pdf"))
        documents: List[dict] = []

        for pdf_file in pdf_files:
            try:
                page_documents = self.load_pdf(pdf_file)
                if not page_documents:
                    continue

                chunked_documents = self.split_documents(page_documents)
                documents.extend(chunked_documents)
            except FileNotFoundError:
                raise
            except Exception:
                logger.exception("Skipping PDF due to error: %s", pdf_file.name)

        return documents
