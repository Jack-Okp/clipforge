"""Run from the backend folder:  python -m pytest tests -q"""
import json

import httpx
import pytest

from app.pipeline import highlights
from app.providers import PROVIDERS, LLMError, get_provider
from app.providers.base import LLMProvider

SEGMENTS = [{"start": i * 10.0, "end": i * 10.0 + 9.5, "text": f"sentence {i}", "words": []} for i in range(30)]


# ---- JSON parsing: models love to wrap or decorate their answers ----

def test_parse_plain_json():
    assert highlights.parse_json('{"clips": [{"start": 1}]}')["clips"][0]["start"] == 1


def test_parse_fenced_json():
    assert highlights.parse_json('```json\n{"clips": []}\n```') == {"clips": []}


def test_parse_json_with_chatter():
    out = highlights.parse_json('Sure! Here you go:\n{"clips": [{"start": 2}]}\nHope that helps.')
    assert out["clips"][0]["start"] == 2


def test_parse_bare_list():
    assert highlights.parse_json('[{"start": 3}]')["clips"][0]["start"] == 3


def test_parse_garbage_raises():
    with pytest.raises(LLMError):
        highlights.parse_json("no json here at all")


# ---- validation ----

def test_validate_snaps_and_clamps():
    raw = [{"start": 12, "end": 52, "title": "t", "score": 150, "reason": "r"}]
    out = highlights.validate_clips(raw, SEGMENTS, 300, 20, 90)
    assert out[0]["start"] == 10.0          # snapped to a sentence start
    assert out[0]["end"] == 49.5            # snapped to a sentence end
    assert out[0]["score"] == 100           # clamped


def test_validate_rejects_bad_lengths_and_junk():
    raw = [
        {"start": 0, "end": 5, "title": "too short", "score": 50},
        {"start": 0, "end": 290, "title": "too long", "score": 50},
        {"start": "x", "end": 9},
        {"start": 50, "end": 40},
    ]
    assert highlights.validate_clips(raw, SEGMENTS, 300, 20, 90) == []


def test_dedupe_keeps_higher_score():
    clips = [
        {"start": 0, "end": 40, "score": 60, "title": "a", "reason": ""},
        {"start": 30, "end": 70, "score": 90, "title": "b", "reason": ""},
        {"start": 100, "end": 140, "score": 50, "title": "c", "reason": ""},
    ]
    out = highlights.dedupe(clips)
    assert [c["title"] for c in out] == ["b", "c"]


def test_chunking_overlaps_and_covers_everything():
    chunks = highlights.chunk_segments(SEGMENTS, max_chars=200, overlap=2)
    assert len(chunks) > 1
    seen = {s["start"] for c in chunks for s in c}
    assert seen == {s["start"] for s in SEGMENTS}


# ---- provider switching ----

CFG = {"llm_provider": "claude", "anthropic_api_key": "k", "anthropic_model": "m",
       "gemini_api_key": "g", "gemini_model": "gm", "ollama_base_url": "http://localhost:11434",
       "ollama_model": "llama3.1:8b"}


def test_registry_and_switching():
    assert set(PROVIDERS) == {"claude", "gemini", "ollama"}
    assert get_provider(CFG).name == "claude"
    assert get_provider({**CFG, "llm_provider": "gemini"}).name == "gemini"
    assert get_provider(CFG, "ollama").name == "ollama"   # per-project override wins
    with pytest.raises(LLMError):
        get_provider(CFG, "nope")


def test_missing_keys_give_clear_errors():
    with pytest.raises(LLMError, match="Anthropic API key"):
        get_provider({**CFG, "anthropic_api_key": ""}).generate("s", "p")
    with pytest.raises(LLMError, match="Gemini API key"):
        get_provider({**CFG, "gemini_api_key": ""}, "gemini").generate("s", "p")


def _mock(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    real_post = httpx.post
    monkeypatch.setattr(httpx, "post", lambda url, **kw: httpx.Client(transport=transport).post(url, **kw))
    return real_post


def test_claude_request_and_parse(monkeypatch):
    seen = {}

    def handler(req):
        seen["headers"], seen["body"] = req.headers, json.loads(req.content)
        return httpx.Response(200, json={"content": [{"type": "text", "text": '{"ok": true}'}]})

    _mock(monkeypatch, handler)
    assert get_provider(CFG).generate("sys", "hi") == '{"ok": true}'
    assert seen["headers"]["x-api-key"] == "k"
    assert seen["body"]["model"] == "m"


def test_gemini_request_and_parse(monkeypatch):
    seen = {}

    def handler(req):
        seen["url"], seen["body"] = str(req.url), json.loads(req.content)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "hello"}]}}]})

    _mock(monkeypatch, handler)
    assert get_provider(CFG, "gemini").generate("sys", "hi") == "hello"
    assert "gm:generateContent" in seen["url"]
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"


def test_ollama_request_and_parse(monkeypatch):
    def handler(req):
        body = json.loads(req.content)
        assert body["format"] == "json" and body["stream"] is False
        return httpx.Response(200, json={"message": {"content": "local"}})

    _mock(monkeypatch, handler)
    assert get_provider(CFG, "ollama").generate("sys", "hi") == "local"


def test_http_errors_become_llm_errors(monkeypatch):
    _mock(monkeypatch, lambda req: httpx.Response(401, json={"error": {"message": "bad key"}}))
    with pytest.raises(LLMError, match="bad key"):
        get_provider(CFG).generate("s", "p")


# ---- end to end highlight finding with a fake provider ----

class Fake(LLMProvider):
    name = "fake"

    def generate(self, system, prompt, json_mode=True, max_tokens=4096):
        return json.dumps({"clips": [
            {"start": 10, "end": 49.5, "title": "good", "score": 88, "reason": "r"},
            {"start": 100, "end": 139.5, "title": "also good", "score": 70, "reason": "r"},
        ]})


def test_find_highlights_end_to_end():
    cfg = {"min_clip_seconds": "20", "max_clip_seconds": "90", "max_clips": "5"}
    out = highlights.find_highlights(Fake({}), {"segments": SEGMENTS, "duration": 300}, cfg)
    assert [c["title"] for c in out] == ["good", "also good"]
