"""LLM answer generation using Google Gemini.

This module manages Gemini client initialization, prompt composition, and
answer generation using the Google Generative AI Python client.
"""

import logging
from pathlib import Path
from typing import Any

import google.generativeai as gen

from app import config

logger = logging.getLogger(__name__)


class GeminiClient:
    """Wrapper for Gemini answer generation."""

    _client_initialized = False

    @classmethod
    def initialize_client(cls) -> None:
        """Initialize the Gemini client using the configured API key."""
        api_key = config.GOOGLE_API_KEY
        if not isinstance(api_key, str) or not api_key.strip():
            logger.error("Google API key is not configured for Gemini.")
            raise ValueError("Google API key is required for Gemini integration.")

        if cls._client_initialized:
            return

        try:
            gen.configure(api_key=api_key)
            cls._client_initialized = True
            logger.info("Gemini client configured successfully.")
        except Exception as exc:
            logger.exception("Failed to configure Gemini client.")
            raise RuntimeError("Failed to initialize the Gemini client.") from exc

    @staticmethod
    def build_prompt(context: str, question: str) -> str:
        """Build the Gemini prompt from the security prompt template."""
        if not isinstance(question, str) or not question.strip():
            logger.error("build_prompt called with empty question")
            raise ValueError("Question text cannot be empty.")

        prompt_path = config.SECURITY_PROMPT_PATH
        if not isinstance(prompt_path, Path) or not prompt_path.exists():
            logger.error("Security prompt template not found: %s", prompt_path)
            raise FileNotFoundError("Security prompt template could not be loaded.")

        prompt_template = prompt_path.read_text(encoding="utf-8")
        prompt = prompt_template.replace("{context}", context or "")
        prompt = prompt.replace("{question}", question.strip())

        logger.info("Gemini prompt built")
        return prompt

    @classmethod
    def generate_answer(cls, question: str, context: str) -> str:
        """Generate an answer from Gemini given the question and retrieved context."""
        cls.initialize_client()

        prompt = cls.build_prompt(context or "", question)

        try:
            model = gen.GenerativeModel(
                model_name=config.LLM_MODEL_NAME,
                generation_config={"temperature": config.GEMINI_TEMPERATURE},
            )
            response = model.generate_content(prompt)
        except Exception as exc:
            logger.exception("Gemini answer generation failed")
            raise RuntimeError("Gemini answer generation failed.") from exc

        answer = getattr(response, "text", None)
        if not answer:
            candidates = getattr(response, "candidates", None)
            if candidates:
                first_candidate = candidates[0]
                answer = getattr(first_candidate, "content", None)

        if not isinstance(answer, str) or not answer.strip():
            logger.error("Gemini returned an empty response")
            raise RuntimeError("Gemini returned an empty or invalid response.")

        logger.info("Gemini answer generated successfully")
        return answer.strip()
