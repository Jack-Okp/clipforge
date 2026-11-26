"""Local models through an Ollama server (no API key, fully offline)."""
import httpx

from .base import LLMError, LLMProvider


class OllamaProvider(LLMProvider):
    name = "ollama"
    label = "Ollama (local)"

    @property
    def base(self) -> str:
        return (self.cfg.get("ollama_base_url") or "http://localhost:11434").rstrip("/")

    def generate(self, system: str, prompt: str, json_mode: bool = True, max_tokens: int = 4096) -> str:
        model = self.cfg.get("ollama_model", "").strip()
        if not model:
            raise LLMError("No Ollama model selected. Pick one in Settings.")
        body: dict = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": {"num_ctx": 16384, "num_predict": max_tokens, "temperature": 0.3},
        }
        if json_mode:
            body["format"] = "json"
        try:
            r = httpx.post(f"{self.base}/api/chat", json=body, timeout=900)
        except httpx.ConnectError as e:
            raise LLMError(
                f"Cannot connect to Ollama at {self.base}. Is it running? (start the Ollama app or run 'ollama serve')"
            ) from e
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama request failed: {e}") from e
        if r.status_code != 200:
            raise LLMError(f"Ollama error {r.status_code}: {r.text[:300]}")
        text = r.json().get("message", {}).get("content", "")
        if not text:
            raise LLMError("Ollama returned an empty reply.")
        return text


def list_models(base_url: str) -> list[str]:
    """Models already pulled on this machine."""
    base = (base_url or "http://localhost:11434").rstrip("/")
    try:
        r = httpx.get(f"{base}/api/tags", timeout=5)
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise LLMError(f"Cannot reach Ollama at {base}: {e}") from e
    return [m["name"] for m in r.json().get("models", [])]
