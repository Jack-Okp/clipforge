"""Common interface for every LLM backend."""
from abc import ABC, abstractmethod


class LLMError(Exception):
    """Raised with a human-readable message the UI can show directly.

    transient=True marks problems that usually pass on their own (rate limit, overloaded server,
    dropped connection), so the caller may wait and try again. A wrong key or model never is.
    """

    def __init__(self, message: str, transient: bool = False):
        super().__init__(message)
        self.transient = transient


TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504, 529}


class LLMProvider(ABC):
    name: str = "base"
    label: str = "Base"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def generate(self, system: str, prompt: str, json_mode: bool = True, max_tokens: int = 4096) -> str:
        """Return the raw text reply. Raise LLMError on any failure."""

    def test(self) -> str:
        """Cheap round trip used by the Settings page's 'Test connection' button."""
        reply = self.generate(
            "You are a connectivity check.",
            'Reply with exactly this JSON and nothing else: {"ok": true}',
            json_mode=True,
            max_tokens=50,
        )
        return reply.strip()[:200]
