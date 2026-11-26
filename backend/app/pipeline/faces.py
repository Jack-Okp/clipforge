"""Face detection, speaker and layout choice, and smooth camera paths.

Uses only the face detectors that ship inside OpenCV, so nothing has to be downloaded.
Faces are found on a handful of frames per second, then turned into a camera path that
moves like a person operating a camera: it ignores small movements, glides when the
speaker really moves, and never jitters.
"""
import json
import math
import subprocess
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

CACHE_VERSION = 2
_cascades: dict = {}


def _cascade(name: str):
    if name not in _cascades:
        c = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / name))
        if c.empty():
            raise RuntimeError(f"OpenCV could not load {name}")
        _cascades[name] = c
    return _cascades[name]


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter) if inter else 0.0


def _nms(boxes: list[tuple], thr: float = 0.3) -> list[tuple]:
    kept: list[tuple] = []
    for b in sorted(boxes, key=lambda b: -b[2] * b[3]):
        if all(_iou(b, k) < thr for k in kept):
            kept.append(b)
    return kept


def detect_in_frame(gray: np.ndarray, min_size: int) -> list[tuple]:
    """Frontal faces first; if none, try both profile directions."""
    g = cv2.equalizeHist(gray)
    ms = (min_size, min_size)
    boxes = [tuple(int(v) for v in b)
             for b in _cascade("haarcascade_frontalface_alt2.xml").detectMultiScale(g, 1.1, 5, minSize=ms)]
    if not boxes:
        prof = _cascade("haarcascade_profileface.xml")
        boxes = [tuple(int(v) for v in b) for b in prof.detectMultiScale(g, 1.1, 5, minSize=ms)]
        width = g.shape[1]
        for x, y, w, h in prof.detectMultiScale(cv2.flip(g, 1), 1.1, 5, minSize=ms):
            boxes.append((int(width - x - w), int(y), int(w), int(h)))
    return _nms(boxes)


def _roi(gray: np.ndarray, box, y0: float, y1: float, x0: float = 0.2, x1: float = 0.8):
    x, y, w, h = box
    gh, gw = gray.shape
    xa, xb = max(int(x + x0 * w), 0), min(int(x + x1 * w), gw)
    ya, yb = max(int(y + y0 * h), 0), min(int(y + y1 * h), gh)
    if xb - xa < 4 or yb - ya < 4:
        return None
    return cv2.resize(gray[ya:yb, xa:xb], (24, 12), interpolation=cv2.INTER_AREA).astype(np.float32)


def mouth_activity(gray, prev_gray, box, prev_boxes) -> float:
    """How much the mouth area changed since the last sample, minus whole-head movement.

    A heuristic, not true lip reading: it is good enough to tell which of two people
    in a shot is talking, which is all the camera needs.
    """
    if prev_gray is None or not prev_boxes:
        return 0.0
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    pb = min(prev_boxes, key=lambda p: math.hypot(p[0] + p[2] / 2 - cx, p[1] + p[3] / 2 - cy))
    if math.hypot(pb[0] + pb[2] / 2 - cx, pb[1] + pb[3] / 2 - cy) > box[2]:
        return 0.0
    m_now, m_prev = _roi(gray, box, 0.62, 0.95), _roi(prev_gray, pb, 0.62, 0.95)
    u_now, u_prev = _roi(gray, box, 0.10, 0.40), _roi(prev_gray, pb, 0.10, 0.40)
    if m_now is None or m_prev is None or u_now is None or u_prev is None:
        return 0.0
    mouth = float(np.abs(m_now - m_prev).mean())
    upper = float(np.abs(u_now - u_prev).mean())
    return max(0.0, mouth - 0.7 * upper)


def stabilise(samples: list[dict], radius: int = 2) -> list[dict]:
    """Drop one-off detections: a real face shows up again within a couple of samples."""
    if len(samples) < 3:
        return samples
    out = []
    for i, s in enumerate(samples):
        keep = []
        for f in s["faces"]:
            for j in range(max(0, i - radius), min(len(samples), i + radius + 1)):
                if j != i and any(abs(g[0] - f[0]) < 0.6 * f[2] and abs(g[1] - f[1]) < 0.6 * f[3]
                                  for g in samples[j]["faces"]):
                    keep.append(f)
                    break
        out.append({"t": s["t"], "faces": keep})
    return out


def scan(src: Path, start: float, dur: float, src_w: int, src_h: int, sample_fps: int = 6,
         work_w: int = 800, check: Callable[[], None] | None = None,
         progress: Callable[[float], None] | None = None) -> list[dict]:
    """Detect faces over a clip. Returns [{t, faces: [[cx, cy, w, h, mouth], ...]}] in source pixels."""
    ww = (min(work_w, src_w) // 2) * 2
    wh = max(2, int(round(src_h * ww / src_w / 2)) * 2)
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
           "-vf", f"fps={sample_fps},scale={ww}:{wh}", "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    expected = max(int(dur * sample_fps), 1)
    min_size, factor = max(24, int(ww * 0.03)), src_w / ww
    samples, prev_gray, prev_boxes, i = [], None, [], 0
    try:
        while True:
            buf = proc.stdout.read(ww * wh)
            if len(buf) < ww * wh:
                break
            if check:
                check()
            gray = np.frombuffer(buf, np.uint8).reshape(wh, ww)
            boxes = detect_in_frame(gray, min_size)
            faces = [[(x + w / 2) * factor, (y + h / 2) * factor, w * factor, h * factor,
                      mouth_activity(gray, prev_gray, (x, y, w, h), prev_boxes)]
                     for x, y, w, h in boxes]
            samples.append({"t": i / sample_fps, "faces": faces})
            prev_gray, prev_boxes, i = gray, boxes, i + 1
            if progress and i % 6 == 0:
                progress(min(i / expected, 1.0))
    finally:
        proc.kill()
        proc.wait()
    return stabilise(samples)


def load_or_scan(cache_file: Path, src: Path, start: float, dur: float, src_w: int, src_h: int,
                 check=None, progress=None) -> list[dict]:
    st = src.stat()
    key = {"v": CACHE_VERSION, "start": round(start, 2), "dur": round(dur, 2), "w": src_w, "h": src_h,
           "size": st.st_size, "mtime": int(st.st_mtime)}
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            if data.get("key") == key:
                return data["samples"]
        except (OSError, ValueError, KeyError):
            pass
    samples = scan(src, start, dur, src_w, src_h, check=check, progress=progress)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"key": key, "samples": samples}))
    return samples


# ---------------------------------------------------------------- layout choice

def _two_faces(s: dict) -> bool:
    f = s["faces"]
    return len(f) >= 2 and max(abs(a[0] - b[0]) for a in f for b in f) > 1.0 * max(a[2] for a in f)


def choose_layout(samples: list[dict], requested: str) -> str:
    n = max(len(samples), 1)
    two = sum(_two_faces(s) for s in samples) / n
    one = sum(bool(s["faces"]) for s in samples) / n
    if requested == "fit":
        return "fit"
    if requested == "fill":
        return "fill"
    if requested == "split":
        return "split" if two >= 0.25 else "fill"
    if two >= 0.5:
        return "split"
    return "fill" if one >= 0.3 else "fit"


# ---------------------------------------------------------------- who to follow

def speaker_series(samples: list[dict]) -> list[tuple | None]:
    """Pick one face per sample: the person who is talking, switching only when it is clear."""
    tracks: list[dict] = []
    current: int | None = None
    streak = 0
    out: list[tuple | None] = []
    for s in samples:
        used: dict[int, list] = {}
        for f in sorted(s["faces"], key=lambda f: -f[2]):
            best, bd = None, 1e18
            for ti, tr in enumerate(tracks):
                d = math.hypot(f[0] - tr["cx"], f[1] - tr["cy"])
                if ti not in used and d < bd and d < 0.9 * max(f[2], tr["w"]):
                    best, bd = ti, d
            if best is None:
                tracks.append({"cx": f[0], "cy": f[1], "w": f[2], "ema": f[4], "miss": 0})
                best = len(tracks) - 1
            else:
                tr = tracks[best]
                tr.update(cx=f[0], cy=f[1], w=f[2], ema=0.6 * tr["ema"] + 0.4 * f[4], miss=0)
            used[best] = f
        for ti, tr in enumerate(tracks):
            if ti not in used:
                tr["miss"] += 1
        if not used:
            out.append(None)
            continue
        if current is None or (current not in used and tracks[current]["miss"] > 6):
            current = max(used, key=lambda t: tracks[t]["w"])
            streak = 0
        elif current in used:
            best = max(used, key=lambda t: tracks[t]["ema"])
            if best != current and tracks[best]["ema"] > tracks[current]["ema"] * 1.6 + 0.8:
                streak += 1
            else:
                streak = 0
            if streak >= 3:
                current, streak = best, 0
        tr = tracks[current]
        out.append((tr["cx"], tr["cy"], tr["w"]))
    return out


def split_series(samples: list[dict]) -> tuple[list, list]:
    """Left-hand and right-hand person for the stacked layout."""
    xs = [f[0] for s in samples if _two_faces(s) for f in s["faces"]]
    if not xs:
        xs = [f[0] for s in samples for f in s["faces"]] or [0.0, 1.0]
    c0, c1 = min(xs), max(xs)
    for _ in range(8):
        a = [x for x in xs if abs(x - c0) <= abs(x - c1)]
        b = [x for x in xs if abs(x - c0) > abs(x - c1)]
        c0 = sum(a) / len(a) if a else c0
        c1 = sum(b) / len(b) if b else c1
    left, right = [], []
    for s in samples:
        lf = rf = None
        for f in sorted(s["faces"], key=lambda f: -f[2]):
            if abs(f[0] - c0) <= abs(f[0] - c1):
                lf = lf or (f[0], f[1], f[2])
            else:
                rf = rf or (f[0], f[1], f[2])
        left.append(lf)
        right.append(rf)
    return left, right


# ---------------------------------------------------------------- camera path

def _fill_gaps(vals: list) -> np.ndarray | None:
    """Hold the last known value through gaps, and back-fill any gap at the start."""
    known = [v for v in vals if v is not None]
    if not known:
        return None
    out, last = [], known[0]
    for v in vals:
        last = v if v is not None else last
        out.append(last)
    return np.array(out, dtype=np.float64)


def _median(a: np.ndarray, k: int = 5) -> np.ndarray:
    if len(a) < k:
        return a
    pad = np.pad(a, k // 2, mode="edge")
    return np.array([np.median(pad[i:i + k]) for i in range(len(a))])


def lazy_follow(target: np.ndarray, fps: float, dead: float, vmax: float, ease: float = 0.18) -> np.ndarray:
    """A camera that stays put inside a dead zone and eases toward the subject outside it."""
    cam = float(target[0])
    out = np.empty_like(target)
    step_max = vmax / fps
    for i, tgt in enumerate(target):
        d = tgt - cam
        if abs(d) > dead:
            goal = tgt - math.copysign(dead * 0.5, d)
            cam += max(-step_max, min(step_max, (goal - cam) * ease))
        out[i] = cam
    return out


def gaussian(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < 0.5 or len(a) < 3:
        return a
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(a, r, mode="edge"), k, mode="valid")


def camera_path(ts: list[float], vals: list, n_frames: int, fps: float,
                dead: float, vmax: float) -> np.ndarray | None:
    """One smooth coordinate per output frame, or None when nothing was ever detected."""
    arr = _fill_gaps(vals)
    if arr is None:
        return None
    arr = _median(arr, 5)
    frames = np.interp(np.arange(n_frames) / fps, ts, arr)
    return gaussian(lazy_follow(frames, fps, dead, vmax), 0.3 * fps)
