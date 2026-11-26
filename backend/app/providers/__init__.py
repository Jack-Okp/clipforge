"""Provider registry. Add a new backend by subclassing LLMProvider and listing it here."""
from .base import LLMError, LLMProvider
from .claude import ClaudeProvider
from .gemini import GeminiProvider
from .ollama import OllamaProvider

PROVIDERS: dict[str, type[LLMProvider]] = {
    "claude": ClaudeProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
}


def get_provider(cfg: dict, name: str | None = None) -> LLMProvider:
    """Build the provider chosen in settings (or an explicit override)."""
    name = name or cfg.get("llm_provider", "claude")
    if name not in PROVIDERS:
        raise LLMError(f"Unknown provider '{name}'. Choose one of: {', '.join(PROVIDERS)}")
    return PROVIDERS[name](cfg)


__all__ = ["PROVIDERS", "get_provider", "LLMError", "LLMProvider"]
