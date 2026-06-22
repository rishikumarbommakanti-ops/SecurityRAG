"""Hybrid answer engine for SecurityRAG.

This module decides whether to answer locally using deterministic pattern
matching or to delegate to Gemini when the retrieval confidence is low.
"""

import logging
import re
from enum import Enum
from typing import List

from app import config
from app.llm.gemini_client import GeminiClient
from app.retrieval.retriever import SecurityRetriever

logger = logging.getLogger(__name__)


class AnswerSource(str, Enum):
    LOCAL = "local"
    GEMINI = "gemini"


class AnswerEngine:
    """Hybrid answer engine for deciding the best answer path."""

    KNOWN_PATTERN_PATTERNS = [
        re.compile(r"T\d{4}", re.IGNORECASE),
        re.compile(r"Event\s*ID\s*\d+", re.IGNORECASE),
        re.compile(r"Sigma\b", re.IGNORECASE),
    ]

    def __init__(self, retriever: SecurityRetriever, confidence_threshold: float = 0.8) -> None:
        self.retriever = retriever
        self.confidence_threshold = confidence_threshold

    def can_answer_locally(self, documents: List[dict]) -> bool:
        """Return True when the engine can answer deterministically without Gemini."""
        if not documents:
            logger.debug("can_answer_locally: no documents available")
            return False

        top_score = max((doc.get("score", 0.0) for doc in documents), default=0.0)
        if top_score < self.confidence_threshold:
            logger.debug("can_answer_locally: confidence %.3f below threshold %.3f", top_score, self.confidence_threshold)
            return False

        concatenated_text = " ".join(
            str(doc.get("content", "")) + " " + str(doc.get("metadata", {}))
            for doc in documents
        )
        for pattern in self.KNOWN_PATTERN_PATTERNS:
            if pattern.search(concatenated_text):
                logger.debug("can_answer_locally: known security pattern matched: %s", pattern.pattern)
                return True

        logger.debug("can_answer_locally: no known patterns matched despite high confidence")
        return False

    def generate_local_answer(self, documents: List[dict]) -> str:
        """Generate a deterministic local answer from high-confidence documents."""
        if not documents:
            logger.error("generate_local_answer called with no documents")
            raise ValueError("No documents available for local answer generation.")

        top_document = max(documents, key=lambda doc: doc.get("score", 0.0))
        source = top_document.get("metadata", {}).get("source", "Unknown source")
        content = top_document.get("content", "").strip()

        answer = (
            f"I found a high-confidence security match from {source}. "
            f"Based on the retrieved context, here is the most relevant information:\n\n{content}"
        )
        logger.info("Local answer generated from source %s", source)
        return answer

    def generate_ai_answer(self, question: str, documents: List[dict]) -> str:
        """Generate an answer using Gemini when local confidence is insufficient."""
        if not documents:
            logger.error("generate_ai_answer called with no documents")
            raise ValueError("No retrieval context available for Gemini answer generation.")

        if config.DEMO_MODE:
            logger.warning("Demo mode active: skipping Gemini generation.")
            return (
                "Demo mode is enabled. Gemini answer generation is disabled, "
                "so this response is based on available demo knowledge and retrieval context."
            )

        try:
            context = self.retriever.format_context(documents)
            answer = GeminiClient.generate_answer(question, context)
            logger.info("AI answer generated via Gemini")
            return answer
        except Exception as exc:
            logger.exception("Gemini generation failed")
            if config.ENABLE_GEMINI_FALLBACK or config.DEMO_MODE:
                return (
                    "Gemini is currently unavailable. "
                    "Please try again later or enable a valid Gemini API key."
                )
            raise

    def answer(self, question: str, documents: List[dict]) -> dict:
        """Return the best answer and its source type."""
        if self.can_answer_locally(documents):
            return {
                "answer": self.generate_local_answer(documents),
                "answer_source": AnswerSource.LOCAL.value,
            }

        return {
            "answer": self.generate_ai_answer(question, documents),
            "answer_source": AnswerSource.GEMINI.value,
        }
