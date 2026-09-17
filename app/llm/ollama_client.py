
"""
Ollama-Provider - lokale LLM-Inferenz.
"""
import time
import logging

import httpx

from app.llm.provider import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    def __init__(self, host: str, model: str, timeout: float = 600.0):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        start = time.perf_counter()

        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 200,
            },
        }
        if system:
            payload["system"] = system

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.host}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()

        latency_ms = int((time.perf_counter() - start) * 1000)

        return LLMResponse(
            text=data.get("response", "").strip(),
            model=self.model,
            provider="ollama",
            latency_ms=latency_ms,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )

    def health(self) -> bool:
        try:
            with httpx.Client(timeout=3.0) as client:
                response = client.get(f"{self.host}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.warning("Ollama health check failed: %s", e)
            return False
