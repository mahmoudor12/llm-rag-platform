"""
Provider-Factory. Wählt anhand von `settings.llm_provider` den
richtigen Provider — der Rest des Codes bleibt unverändert.
"""
from app.config import settings
from app.llm.provider import LLMProvider
from app.llm.ollama_client import OllamaProvider
from app.llm.anthropic_client import AnthropicProvider


def create_provider() -> LLMProvider:
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        return OllamaProvider(
            host=settings.ollama_host,
            model=settings.llm_model,
        )

    if provider == "anthropic":
        return AnthropicProvider(
            api_key=settings.anthropic_api_key or "",
            model=settings.llm_model,
        )

    raise ValueError(
        f"Unbekannter LLM_PROVIDER: '{provider}'. "
        "Erlaubte Werte: 'ollama', 'anthropic'."
    )