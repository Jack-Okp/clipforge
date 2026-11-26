"""Which parts of a clip survive cleanup, and how source times map onto the edited clip.

Fillers ("um", "uh") and long pauses are cut out using the word timestamps. Every cut is
snapped to the video frame grid so the picture and the sound stay in sync, and captions
are re-timed through Timeline.map_time so they match the edited result.
"""
import re

import numpy as np

# Only sounds that are never real words. "like" and "you know" are left alone on purpose.
FILLERS = {"um", "umm", "uhm", "uh", "uhh", "er", "erm", "ah", "hmm", "hm", "mm", "mmm", "mhm", "eh", "uhhuh"}


def is_filler(word: str) -> bool:
    return re.sub(r"[^a-z]", "", word.lower()) in FILLERS


def clip_words(transcript: dict, start: float, end: float) -> list[dict]:
    """Words of the transcript that fall inside [start, end], in source time."""
    out: list[dict] = []
    for seg in transcript.get("segments", []):
        if seg["end"] <= start or seg["start"] >= end:
            continue
        words = seg.get("words") or _spread_words(seg)
        for w in words:
            text = str(w["word"]).strip()
            if not text or w["end"] <= start or w["start"] >= end:
                continue
            # "key" is the word's original start time: a stable id the editor uses to edit or cut this word
            out.append({"start": max(w["start"], start), "end": min(w["end"], end), "word": text,
                        "key": f"{w['start']:.2f}"})
    return out


def apply_text_edits(words: list[dict], text_edits: dict | None) -> list[dict]:
    """Editor changes to the caption text. Key = the word's original start time; "" hides the word."""
    if not text_edits:
        return words
    out = []
    for w in words:
        t = text_edits.get(w.get("key"))
        if t is None:
            out.append(w)
        elif str(t).strip():
            out.append({**w, "word": str(t).strip()})
    return out


def _spread_words(seg: dict) -> list[dict]:
    """Fallback when a segment has no word timestamps: spread its words by character count."""
    toks = seg["text"].split()
    if not toks:
        return []
    total = sum(len(t) for t in toks)
    span, t, out = seg["end"] - seg["start"], seg["start"], []
    for tok in toks:
        d = span * len(tok) / total
        out.append({"start": t, "end": t + d, "word": tok})
        t += d
    return out


def _merge(cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for a, b in sorted(c for c in cuts if c[1] > c[0]):
        if merged and a <= merged[-1][1] + 0.02:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


class Timeline:
    def __init__(self, start: float, fps: float, total_frames: int, ranges: list[tuple[int, int]]):
        self.start, self.fps, self.total_frames, self.ranges = start, fps, total_frames, ranges
        self.mask = np.zeros(max(total_frames, 1), dtype=bool)
        for fa, fb in ranges:
            self.mask[fa:fb] = True
        self.out_frames = int(self.mask.sum())

    @property
    def out_duration(self) -> float:
        return self.out_frames / self.fps

    @property
    def removed_seconds(self) -> float:
        return (self.total_frames - self.out_frames) / self.fps

    def audio_intervals(self) -> list[tuple[float, float]]:
        """Kept spans in seconds relative to the clip start."""
        return [(fa / self.fps, fb / self.fps) for fa, fb in self.ranges]

    def contains(self, t_src: float) -> bool:
        i = int((t_src - self.start) * self.fps)
        return 0 <= i < len(self.mask) and bool(self.mask[i])

    def map_time(self, t_src: float) -> float:
        """Source time to edited-clip time. A time inside a cut snaps to the next kept moment."""
        f = (t_src - self.start) * self.fps
        cum = 0
        for fa, fb in self.ranges:
            if f < fa:
                return cum / self.fps
            if f <= fb:
                return (cum + f - fa) / self.fps
            cum += fb - fa
        return cum / self.fps

    def remap_words(self, words: list[dict]) -> list[dict]:
        out = []
        for w in words:
            if not self.contains((w["start"] + w["end"]) / 2):
                continue  # word sits inside a removed part
            s = self.map_time(w["start"])
            e = max(self.map_time(w["end"]), s + 0.05)
            out.append({**w, "start": s, "end": e})
        return out


def full_timeline(start: float, end: float, fps: float) -> Timeline:
    n = max(int((end - start) * fps), 1)
    return Timeline(start, fps, n, [(0, n)])


def compute_timeline(start: float, end: float, words: list[dict], fps: float,
                     remove_fillers: bool, remove_silence: bool,
                     silence_gap: float = 0.8, pad: float = 0.18, min_keep: float = 0.12,
                     extra_cuts: list[tuple[float, float]] | None = None) -> Timeline:
    total = max(int((end - start) * fps), 1)
    cuts: list[tuple[float, float]] = list(extra_cuts or [])      # cuts the user made in the editor

    if remove_fillers:
        cuts += [(w["start"], w["end"]) for w in words if is_filler(w["word"])]

    if remove_silence and words:
        speech = [w for w in words if not (remove_fillers and is_filler(w["word"]))]
        prev_end, first = start, True
        for w in speech + [{"start": end, "end": end, "word": "", "_last": True}]:
            gap = w["start"] - prev_end
            if gap > silence_gap:
                left = 0.0 if first else pad            # no padding against the clip's own edge
                right = 0.0 if w.get("_last") else pad
                cuts.append((prev_end + left, w["start"] - right))
            prev_end, first = max(prev_end, w["end"]), False

    if not cuts:
        return full_timeline(start, end, fps)

    keep, cur = [], start
    for a, b in _merge(cuts):
        if a > cur:
            keep.append((cur, a))
        cur = max(cur, b)
    if cur < end:
        keep.append((cur, end))

    ranges: list[tuple[int, int]] = []
    min_frames = max(3, int(min_keep * fps))
    for a, b in keep:
        fa = min(max(round((a - start) * fps), 0), total)
        fb = min(max(round((b - start) * fps), 0), total)
        if fb - fa < min_frames:
            continue
        if ranges and fa <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], fb))
        else:
            ranges.append((fa, fb))
    if not ranges:
        return full_timeline(start, end, fps)
    return Timeline(start, fps, total, ranges)
