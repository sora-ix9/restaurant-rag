"""Qwen LLM service: generates grounded answers using Qwen models via DashScope API."""
import logging
import os
from typing import Optional

import httpx

from app.config import settings
from app.services.prompts import build_messages

logger = logging.getLogger(__name__)


class QwenService:
    """Service for generating answers using Qwen models via DashScope API."""

    def __init__(self):
        # Settings first, env as fallback — both are documented in .env.example,
        # and before this they had no effect at all (drift 2). Keeping os.getenv
        # means nobody's working DASHSCOPE_API_KEY setup breaks.
        self.api_key = settings.qwen_api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = (
            settings.qwen_base_url
            or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        self.model = settings.qwen_model
        self.client: Optional[httpx.Client] = None

        if not self.api_key:
            logger.warning(
                "DASHSCOPE_API_KEY not set. Qwen service will fail on generate_answer()."
            )

    def _get_client(self) -> httpx.Client:
        """Lazy-initialize HTTP client."""
        if self.client is None:
            self.client = httpx.Client(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
        return self.client

    def generate_answer(self, question: str, context: str, prompt: str | None = None) -> str:
        """Generate a grounded answer using Qwen model.

        Args:
            question: User's question
            context: Retrieved document chunks joined together
            prompt: Named prompt set. Falls back to `settings.prompt_set` (D11).

        Returns:
            Generated answer text

        Raises:
            ValueError: If DASHSCOPE_API_KEY is not set
            httpx.HTTPError: If API call fails
        """
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY not set in environment variables")

        messages = build_messages(question, context, prompt)

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 0.9,
        }

        logger.info(f"Calling Qwen API with model={self.model}")
        client = self._get_client()

        response = client.post("/chat/completions", json=payload)
        response.raise_for_status()

        result = response.json()
        answer = result["choices"][0]["message"]["content"]

        logger.info(f"Qwen API returned {len(answer)} chars")
        return answer

    def health_check(self) -> bool:
        """Check if Qwen API is accessible."""
        try:
            self.generate_answer("What is 2+2?", "Basic arithmetic.")
            return True
        except Exception as e:
            logger.error(f"Qwen health check failed: {e}")
            return False


# Singleton instance
qwen_service = QwenService()
