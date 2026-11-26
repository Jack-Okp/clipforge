"""Ask the chosen LLM for the best self-contained clips, then validate and clean the answer.

The model never gets to decide anything unchecked: every clip is clamped to the video,
snapped to sentence boundaries, length-checked, and de-duplicated before it is accepted.
"""
import json
import re
import time

from ..providers.base import LLMError, LLMProvider

SYSTEM = (
    "You are an expert short-form video editor who finds viral moments in long videos "
    "for TikTok, Reels and YouTube Shorts."
)

PROMPT = """Below is a timestamped transcript. Each line starts with [start-end] in seconds.

Pick up to {max_clips} clips. Each clip must:
- be a self-contained moment with a hook at the start and a payoff or clear ending
- start at the beginning of a sentence and end at the end of a sentence
- last between {min_s} and {max_s} seconds
- not overlap another clip

Score each clip 0-100 for viral potential (hook strength, emotion, insight, story, shareability).
Use only timestamps that appear in the transcript.
{focus_block}
Return JSON in exactly this shape:
{{"clips": [{{"start": 12.5, "end": 58.0, "title": "short catchy title", "score": 87, "reason": "one sentence on why it works"}}]}}

TRANSCRIPT:
{transcript}
"""


def segments_to_lines(segments: list[dict]) -> list[str]:
    return [f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments]


def chunk_segments(segments: list[dict], max_chars: int, overlap: int = 6) -> list[list[dict]]:
    """Split a long transcript into overlapping windows that fit a model's context."""
    chunks, cur, size = [], [], 0
    for seg in segments:
        n = len(seg["text"]) + 24
        if cur and size + n > max_chars:
            chunks.append(cur)
            cur = cur[-overlap:]
            size = sum(len(s["text"]) + 24 for s in cur)
        cur.append(seg)
        size += n
    if cur:
        chunks.append(cur)
    return chunks


def parse_json(text: str) -> dict:
    """Tolerant JSON extraction: strips code fences and surrounding chatter."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return _wrap(json.loads(text))
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        a, b = text.find(open_c), text.rfind(close_c)
        if a != -1 and b > a:
            try:
                return _wrap(json.loads(text[a:b + 1]))
            except json.JSONDecodeError:
                continue
    raise BadReply("The model did not return valid JSON.")


def _wrap(obj) -> dict:
    return {"clips": obj} if isinstance(obj, list) else obj


def validate_clips(raw: list[dict], segments: list[dict], duration: float,
                   min_s: float, max_s: float) -> list[dict]:
    """Clamp, snap and filter model output into clips that are safe to cut."""
    starts = [s["start"] for s in segments]
    ends = [s["end"] for s in segments]
    out = []
    for c in raw:
        try:
            start, end = float(c["start"]), float(c["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if start >= end:
            continue
        if starts:  # snap to the nearest sentence boundary
            start = min(starts, key=lambda t: abs(t - start))
            end = min(ends, key=lambda t: abs(t - end))
        start, end = max(0.0, start), min(duration, end)
        length = end - start
        if length < min_s * 0.8 or length > max_s * 1.2:
            continue
        try:
            score = max(0, min(100, int(float(c.get("score", 50)))))
        except (TypeError, ValueError):
            score = 50
        out.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "title": str(c.get("title") or "Untitled clip")[:120],
            "score": score,
            "reason": str(c.get("reason") or "")[:400],
        })
    return dedupe(out)


def dedupe(clips: list[dict]) -> list[dict]:
    """Drop overlapping clips, keeping the higher-scoring one."""
    kept: list[dict] = []
    for c in sorted(clips, key=lambda x: -x["score"]):
        if all(c["end"] <= k["start"] or c["start"] >= k["end"] for k in kept):
            kept.append(c)
    return sorted(kept, key=lambda x: -x["score"])


FOCUS_BLOCK = """
FOCUS: the person only wants moments about: "{focus}".
Choose only moments where that subject is actually discussed. If fewer than {max_clips} qualify, return
only those; never pad the list with unrelated moments. If nothing in this part matches, return {{"clips": []}}.
"""

_sleep = time.sleep           # replaced in tests so retries do not really wait
RETRY_DELAYS = (3, 8)         # seconds to wait before the 2nd and 3rd attempt at a temporary failure


def clean_focus(focus: str | None) -> str:
    """The topic typed by the person: one short line, no quotes that could break the prompt."""
    return re.sub(r"\s+", " ", (focus or "").replace('"', "'")).strip()[:200]


class BadReply(LLMError):
    """The model answered, but not with JSON we can use."""


def _ask(provider: LLMProvider, prompt: str) -> dict:
    """One model call with the retries it deserves:
    - temporary trouble (rate limit, overload, dropped connection): wait and try again, up to 3 attempts
    - unusable JSON: try once more straight away
    - anything else (wrong key, wrong model, blocked): fail at once, retrying cannot help
    """
    waits = list(RETRY_DELAYS)
    bad_replies = 0
    while True:
        try:
            return parse_json(provider.generate(SYSTEM, prompt, json_mode=True, max_tokens=4096))
        except BadReply:
            bad_replies += 1
            if bad_replies >= 2:
                raise
        except LLMError as e:
            if e.transient and waits:
                _sleep(waits.pop(0))
            else:
                raise


def find_highlights(provider: LLMProvider, transcript: dict, cfg: dict, focus: str | None = None) -> list[dict]:
    focus = clean_focus(focus)
    segments = transcript["segments"]
    duration = float(transcript.get("duration") or (segments[-1]["end"] if segments else 0))
    min_s, max_s = float(cfg["min_clip_seconds"]), float(cfg["max_clip_seconds"])
    max_clips = int(cfg["max_clips"])
    if not segments:
        raise LLMError("The transcript is empty, nothing to analyse.")

    # Local models have far smaller usable context than the cloud ones.
    window = 12000 if provider.name == "ollama" else 90000
    raw: list[dict] = []
    for chunk in chunk_segments(segments, window):
        per_chunk = max_clips if len(segments) == len(chunk) else max(3, max_clips // 2)
        prompt = PROMPT.format(
            max_clips=per_chunk,
            min_s=int(min_s), max_s=int(max_s),
            focus_block=FOCUS_BLOCK.format(focus=focus, max_clips=per_chunk) if focus else "",
            transcript="\n".join(segments_to_lines(chunk)),
        )
        raw.extend(_ask(provider, prompt).get("clips", []))

    clips = validate_clips(raw, segments, duration, min_s, max_s)[:max_clips]
    if not clips:
        if focus:
            raise LLMError(f'No moments about "{focus}" were found in this video. '
                           "Try different words, or clear the topic to get the best moments overall.")
        raise LLMError(
            "The model returned no usable clips. Try another model in Settings or adjust the clip length range."
        )
    return clips
