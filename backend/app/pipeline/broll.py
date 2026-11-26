"""B-roll suggestions from the user's own footage folder.

Matching is by words only: file and folder names (e.g. "city_traffic_night.mp4") are compared with
what is said in the clip. There is no visual analysis, so well-named files matter.
"""
import hashlib
import re
import subprocess
from pathlib import Path

from ..config import BROLL_CACHE
from .captions import STOPWORDS

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LIMIT = 3000


def scan(folder: str | Path) -> list[dict]:
    """Media files under the folder (limited), each with name tokens."""
    root = Path(folder)
    if not root.is_dir():
        return []
    items = []
    for p in root.rglob("*"):
        if len(items) >= LIMIT:
            break
        ext = p.suffix.lower()
        if p.is_file() and (ext in VIDEO_EXT or ext in IMAGE_EXT):
            rel = p.relative_to(root)
            items.append({"path": str(p), "name": p.name, "rel": str(rel),
                          "kind": "video" if ext in VIDEO_EXT else "image",
                          "tokens": tokens(" ".join(rel.with_suffix("").parts))})
    return items


def stem(w: str) -> str:
    for suf in ("ing", "ies", "es", "ed", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def tokens(text: str) -> set[str]:
    # split camelCase and separators, drop short and common words
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    out = set()
    for w in re.findall(r"[A-Za-z]{3,}", text.lower()):
        if w not in STOPWORDS:
            out.add(stem(w))
    return out


def is_inside(path: str | Path, folder: str | Path) -> bool:
    """True when path is a real location inside folder (guards against ../ tricks and symlinks)."""
    try:
        return Path(path).resolve().is_relative_to(Path(folder).resolve())
    except (OSError, ValueError):
        return False


def suggest(items: list[dict], words: list[dict], top: int = 12) -> list[dict]:
    """Rank footage against the words of a clip. Each hit says which words matched and where to place it."""
    first_seen: dict[str, float] = {}
    for w in words:
        for t in tokens(w["word"]):
            first_seen.setdefault(t, w["start"])
    scored = []
    for it in items:
        hit = sorted(it["tokens"] & first_seen.keys(), key=lambda t: first_seen[t])
        if hit:
            scored.append({**{k: v for k, v in it.items() if k != "tokens"}, "matched": hit,
                           "score": len(hit), "at": round(first_seen[hit[0]], 2)})
    scored.sort(key=lambda s: (-s["score"], s["at"], s["name"]))
    return scored[:top]


def thumb(path: str | Path) -> Path | None:
    """Small JPEG preview, cached by file identity. None if it cannot be made."""
    p = Path(path)
    if not p.is_file():
        return None
    st = p.stat()
    key = hashlib.sha1(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:20]
    out = BROLL_CACHE / f"{key}.jpg"
    if out.exists():
        return out
    BROLL_CACHE.mkdir(parents=True, exist_ok=True)
    seek = ["-ss", "1"] if p.suffix.lower() in VIDEO_EXT else []
    for attempt in (seek, []):                # very short videos have no frame at 1 s
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", *attempt, "-i", str(p), "-frames:v", "1",
                            "-vf", "scale=240:-2", str(out)], capture_output=True, timeout=60)
        if r.returncode == 0 and out.exists():
            return out
        if not seek:
            break
    return None
