"""Anthropic Claude via the Messages API (plain HTTPS, no SDK needed)."""
import httpx

from .base import TRANSIENT_STATUS, LLMError, LLMProvider

API_URL = "https://api.anthropic.com/v1/messages"


class ClaudeProvider(LLMProvider):
    name = "claude"
    label = "Claude (Anthropic)"

    def generate(self, system: str, prompt: str, json_mode: bool = True, max_tokens: int = 4096) -> str:
        key = self.cfg.get("anthropic_api_key", "").strip()
        if not key:
            raise LLMError("No Anthropic API key set. Add it in Settings.")
        model = self.cfg.get("anthropic_model", "").strip() or "claude-sonnet-5-5"
        if json_mode:
            system += "\n\nRespond with valid JSON only. No markdown fences, no commentary."
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        try:
            r = httpx.post(API_URL, json=body, headers=headers, timeout=180)
        except httpx.HTTPError as e:
            raise LLMError(f"Could not reach Anthropic: {e}", transient=True) from e
        if r.status_code != 200:
            raise LLMError(f"Claude error {r.status_code}: {_err(r)}", transient=r.status_code in TRANSIENT_STATUS)
        parts = r.json().get("content", [])
        text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        if not text:
            raise LLMError("Claude returned an empty reply.")
        return text


def _err(r: httpx.Response) -> str:
    try:
        return r.json().get("error", {}).get("message", r.text[:300])
    except Exception:
        return r.text[:300]
