"""Google Gemini via the Generative Language REST API."""
import httpx

from .base import TRANSIENT_STATUS, LLMError, LLMProvider

BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider(LLMProvider):
    name = "gemini"
    label = "Gemini (Google)"

    def generate(self, system: str, prompt: str, json_mode: bool = True, max_tokens: int = 4096) -> str:
        key = self.cfg.get("gemini_api_key", "").strip()
        if not key:
            raise LLMError("No Gemini API key set. Add it in Settings.")
        model = self.cfg.get("gemini_model", "").strip() or "gemini-3.8-flash"
        gen_cfg: dict = {"maxOutputTokens": max_tokens, "temperature": 0.4}
        if json_mode:
            gen_cfg["responseMimeType"] = "application/json"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": gen_cfg,
        }
        headers = {"x-goog-api-key": key, "content-type": "application/json"}
        try:
            r = httpx.post(f"{BASE}/{model}:generateContent", json=body, headers=headers, timeout=180)
        except httpx.HTTPError as e:
            raise LLMError(f"Could not reach Gemini: {e}", transient=True) from e
        if r.status_code != 200:
            raise LLMError(f"Gemini error {r.status_code}: {_err(r)}", transient=r.status_code in TRANSIENT_STATUS)
        data = r.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError):
            reason = data.get("promptFeedback", {}).get("blockReason", "no candidates")
            raise LLMError(f"Gemini returned no usable reply ({reason}).")
        if not text:
            raise LLMError("Gemini returned an empty reply.")
        return text


def _err(r: httpx.Response) -> str:
    try:
        return r.json().get("error", {}).get("message", r.text[:300])
    except Exception:
        return r.text[:300]
