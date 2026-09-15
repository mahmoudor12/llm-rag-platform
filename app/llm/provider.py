"""
LLM-Provider-Abstraktion.

Jeder Provider implementiert dasselbe Interface. Der Rest des Systems
weiß nichts über Ollama oder Anthropic — nur über `LLMProvider`.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMProvider(ABC):
    """Interface für alle LLM-Provider."""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Erzeugt eine Antwort für den Prompt."""

    @abstractmethod
    def health(self) -> bool:
        """Prüft, ob der Provider erreichbar ist."""