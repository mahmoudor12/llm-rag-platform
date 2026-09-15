"""
Anthropic-Provider — für Demo-Zwecke und schnelle Iteration.

Nutzt die Messages-API von Anthropic.
"""
import time
import logging

import httpx

from app.llm.provider import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    BASE_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4",
        timeout: float = 60.0,
    ):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY ist erforderlich")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        start = time.perf_counter()

        payload: dict = {
            "model": self.model,
            "max_tokens": 512,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.BASE_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        latency_ms = int((time.perf_counter() - start) * 1000)

        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )

        usage = data.get("usage", {})

        return LLMResponse(
            text=text.strip(),
            model=self.model,
            provider="anthropic",
            latency_ms=latency_ms,
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
        )

    def health(self) -> bool:
        return bool(self.api_key)