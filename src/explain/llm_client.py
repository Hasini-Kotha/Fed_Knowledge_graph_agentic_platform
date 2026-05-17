"""LLM Client — Abstracted API wrapper for Ollama.

Supports:
- Chat-style prompts (system + user roles)
- JSON mode for structured output
- Retry with exponential backoff
- Timeout enforcement
- Raw response logging for audit
"""

import json
import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass


class OllamaClient:
    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout_seconds: int = 60,
        max_retries: int = 2,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout_seconds
        self.max_retries = max_retries

    def _build_payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

    def _parse_response(self, raw: str) -> dict[str, Any]:
        """Parse JSON from the LLM response, handling markdown fences and surrounding text."""
        cleaned = raw.strip()

        # Remove markdown code fences if present
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            start = 1 if lines[0].startswith("```") else 0
            end = -1 if lines[-1].strip() == "```" else len(lines)
            cleaned = "\n".join(lines[start:end]).strip()

        # Find the first '{' and last '}' to extract JSON object from surrounding text
        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            cleaned = cleaned[brace_start : brace_end + 1]

        return json.loads(cleaned)

    def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        url = f"{self.base_url}/api/chat"

        for attempt in range(1 + self.max_retries):
            try:
                payload = self._build_payload(messages)
                resp = requests.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()

                raw_content = data.get("message", {}).get("content", "")
                if not raw_content:
                    raise LLMError("Empty response from LLM")

                parsed = self._parse_response(raw_content)

                logger.info(
                    "LLM response: transaction_id=%s, action=%s, confidence=%.2f",
                    parsed.get("transaction_id", "?"),
                    parsed.get("suggested_action", "?"),
                    parsed.get("confidence", 0),
                )
                return parsed

            except json.JSONDecodeError as e:
                logger.warning("JSON parse error (attempt %d/ %d): %s", attempt + 1, self.max_retries + 1, e)
                if attempt >= self.max_retries:
                    raise LLMError(f"Failed to parse LLM response after {self.max_retries + 1} attempts: {e}") from e
                time.sleep(2 ** attempt)

            except requests.RequestException as e:
                logger.warning("Request failed (attempt %d/ %d): %s", attempt + 1, self.max_retries + 1, e)
                if attempt >= self.max_retries:
                    raise LLMError(f"LLM request failed after {self.max_retries + 1} attempts: {e}") from e
                time.sleep(2 ** attempt)

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            available = [m["name"] for m in models]
            logger.info("Ollama available. Models: %s", available)
            return True
        except requests.RequestException as e:
            logger.warning("Ollama health check failed: %s", e)
            return False
