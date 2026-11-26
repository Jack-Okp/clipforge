"""Background job runner. One worker keeps the GPU from being shared by two heavy jobs at once."""
import json
import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import db, settings_store
from .config import PROJECTS_DIR
from .pipeline import export, highlights, ingest, render, transcribe
from .providers import LLMError, get_provider

log = logging.getLogger("clipforge")
_sleep = time.sleep
DOWNLOAD_DELAYS = (4, 12)
_TEMPORARY = ("timed out", "timeout", "connection", "temporarily", "http error 5", "http error 429",
              "too many requests", "unable to download", "reset by peer")

_pool = ThreadPoolExecutor(max_workers=1)
_cancelled: set[str] = set()
_lock = threading.Lock()


class Cancelled(Exception):
    pass


def _msg(e: Exception) -> str:
    """Our own RuntimeError/LLMError texts are already written for people; others get their type name."""
    return str(e) if isinstance(e, (RuntimeError, LLMError)) else f"{type(e).__name__}: {e}"


def recover_interrupted() -> int:
    """Called once at startup. A job cannot survive the app closing, so anything still marked as
    running or queued was cut off. Marking it failed unlocks Run again, Restyle and Delete."""
    n = db.rows("SELECT COUNT(*) AS n FROM projects WHERE status IN ('running', 'queued')")[0]["n"]
    if n:
        db.execute("UPDATE projects SET status = 'failed', error = ? WHERE status IN ('running', 'queued')",
                   ("ClipForge was closed while this was running. Press Run again"
                    " (or Restyle, if clips were already made).",))
    return n


def project_dir(pid: str) -> Path:
    d = PROJECTS_DIR / pid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _update(pid: str, **fields) -> None:
    cols = ", ".join(f"{k} = ?" for k in fields)
    db.execute(f"UPDATE projects SET {cols} WHERE id = ?", (*fields.values(), pid))


def _check_cancel(pid: str) -> None:
    with _lock:
        if pid in _cancelled:
            raise Cancelled()


def cancel(pid: str) -> None:
    with _lock:
        _cancelled.add(pid)


def enqueue(pid: str, provider: str | None = None, from_stage: str = "ingest", focus: str | None = None) -> None:
    with _lock:
        _cancelled.discard(pid)
    extra = {"focus": highlights.clean_focus(focus) or None} if focus is not None else {}
    _update(pid, status="queued", stage=from_stage, progress=0, error=None, queued_at=int(time.time() * 1000), **extra)
    _pool.submit(_run, pid, provider, from_stage)


def enqueue_render(pid: str, clip_ids: list[int] | None, overrides: dict[str, str]) -> None:
    """Restyle existing clips (all, or just some) without touching the transcript or the AI picks."""
    with _lock:
        _cancelled.discard(pid)
    _update(pid, status="queued", stage="render", progress=0, error=None, queued_at=int(time.time() * 1000))
    _pool.submit(_run_render, pid, clip_ids, overrides)


def _stage_progress(pid: str, stage: str, base: float, span: float):
    """Map a 0..1 stage progress onto the project's overall 0..1 progress bar."""
    def cb(frac: float) -> None:
        _check_cancel(pid)
        _update(pid, stage=stage, progress=round(base + span * frac, 3))
    return cb


def _style_cfg(cfg: dict, *layers: dict) -> dict:
    """Later layers win: settings < what the clip was last rendered with < this request."""
    merged = dict(cfg)
    for layer in layers:
        merged.update({k: v for k, v in (layer or {}).items()
                       if k in settings_store.STYLE_KEYS or k == "watermark_path"})
    return merged


def _loads(text, default):
    try:
        return json.loads(text) if text else default
    except ValueError:
        return default


def _download(url: str, pdir: Path, cb):
    """Links fail for passing reasons (network blip, site rate limit), so try up to three times."""
    delays = list(DOWNLOAD_DELAYS)
    while True:
        try:
            return ingest.download_url(url, pdir, cb)
        except Cancelled:
            raise
        except Exception as e:
            if delays and any(t in str(e).lower() for t in _TEMPORARY):
                _check_cancel_sleep(pdir.name, delays.pop(0))
            else:
                raise


def _check_cancel_sleep(pid: str, seconds: float) -> None:
    _sleep(seconds)
    _check_cancel(pid)


def _render_clips(pid: str, rows: list[dict], cfg: dict, overrides: dict, pdir: Path, video: Path,
                  transcript: dict, base: float, span: float) -> list[str]:
    """Render each clip. A failure on one clip is recorded on that clip and the rest carry on."""
    clips_dir = pdir / "clips"
    clips_dir.mkdir(exist_ok=True)
    errors: list[str] = []
    for n, row in enumerate(rows):
        _check_cancel(pid)
        saved = _loads(row.get("opts"), {})
        edits = _loads(row.get("edits"), None) or None
        style = _style_cfg(cfg, saved, overrides)
        fancy = render.needs_render(style) or bool(edits)
        dst = clips_dir / row["file"]
        done_before = n

        def on_prog(frac: float, _n=done_before) -> None:
            _update(pid, stage="render" if fancy else "export",
                    progress=round(base + span * (_n + frac) / len(rows), 3))

        try:
            if fancy:
                info = render.render_clip(video, dst, row, transcript, style, pdir / "work" / f"clip_{row['id']}",
                                          on_prog, lambda: _check_cancel(pid), cache_dir=pdir / "cache", edits=edits)
            else:
                enc = export.cut_clip(video, dst, row["start"], row["end"], style.get("use_nvenc", "auto"))
                info = {"layout": "original", "encoder": enc, "out_seconds": round(row["end"] - row["start"], 2),
                        "removed_seconds": 0}
            keep = {**saved, **{k: v for k, v in overrides.items() if k in settings_store.STYLE_KEYS or k == "watermark_path"}}
            db.execute(
                "UPDATE clips SET layout=?, encoder=?, out_seconds=?, removed_seconds=?, rendered_at=?, opts=?, "
                "words=?, error=NULL WHERE id=?",
                (info["layout"], info["encoder"], info["out_seconds"], info["removed_seconds"],
                 int(time.time()), json.dumps(keep) if keep else None,
                 json.dumps(info["words"]) if info.get("words") is not None else None, row["id"]))
        except Cancelled:
            raise
        except Exception as e:
            msg = _msg(e)
            log.exception("clip %s of project %s failed to render", row["id"], pid)
            errors.append(msg)
            db.execute("UPDATE clips SET error=? WHERE id=?", (msg[:500], row["id"]))
        _update(pid, progress=round(base + span * (n + 1) / len(rows), 3))
    shutil.rmtree(pdir / "work", ignore_errors=True)
    return errors


def _run(pid: str, provider_override: str | None, from_stage: str) -> None:
    pdir = project_dir(pid)
    cfg = settings_store.get_all()
    try:
        _check_cancel(pid)                    # cancelled while still waiting in the queue
        _update(pid, status="running")
        proj = db.one("SELECT * FROM projects WHERE id = ?", (pid,))

        # 1. ingest
        if from_stage == "ingest":
            if proj["source"] and proj["source"].startswith(("http://", "https://")):
                path, title = _download(proj["source"], pdir, _stage_progress(pid, "ingest", 0.0, 0.1))
                _update(pid, title=title)
            else:
                path = pdir / proj["video_file"]
            _update(pid, video_file=path.name, duration=ingest.probe_duration(path), progress=0.1)
            render.probe(path)  # fail now, with a clear message, if there is no video track
        proj = db.one("SELECT * FROM projects WHERE id = ?", (pid,))
        video = pdir / proj["video_file"]
        _check_cancel(pid)

        # 2. transcribe (skipped when re-running highlights with a different model)
        tpath = pdir / "transcript.json"
        if from_stage in ("ingest", "transcribe") or not tpath.exists():
            _update(pid, stage="transcribe")
            transcribe.transcribe(video, tpath, cfg["whisper_model"], cfg["whisper_device"],
                                  _stage_progress(pid, "transcribe", 0.1, 0.4), cfg["transcribe_fillers"] == "on")
        transcript = json.loads(tpath.read_text(encoding="utf-8"))
        _check_cancel(pid)

        # 3. highlights
        _update(pid, stage="highlights", progress=0.5)
        provider = get_provider(cfg, provider_override)
        clips = highlights.find_highlights(provider, transcript, cfg, focus=proj.get("focus"))
        _check_cancel(pid)

        # 4. new clip rows (old clips from a previous run are replaced), then render them
        clips_dir = pdir / "clips"
        if clips_dir.exists():
            shutil.rmtree(clips_dir)
        clips_dir.mkdir()
        shutil.rmtree(pdir / "cache", ignore_errors=True)
        db.execute("DELETE FROM clips WHERE project_id = ?", (pid,))
        for i, c in enumerate(clips, 1):
            db.execute(
                'INSERT INTO clips(project_id, start, "end", title, score, reason, file, provider) '
                "VALUES(?,?,?,?,?,?,?,?)",
                (pid, c["start"], c["end"], c["title"], c["score"], c["reason"], f"clip_{i:02d}.mp4", provider.name))
        rows = db.rows('SELECT id, start, "end", title, file, opts, edits FROM clips WHERE project_id = ? ORDER BY id', (pid,))
        for r in rows:
            r["end"] = float(r["end"])
        errors = _render_clips(pid, rows, cfg, {}, pdir, video, transcript, 0.5, 0.5)

        if len(errors) == len(rows):
            _update(pid, status="failed", error=f"Every clip failed to render. First error: {errors[0]}")
        else:
            note = f"{len(errors)} of {len(rows)} clips failed to render; see the red cards." if errors else None
            _update(pid, status="done", stage="done", progress=1.0, provider_used=provider.name, error=note)
    except Cancelled:
        _update(pid, status="cancelled", error="Cancelled by user")
    except Exception as e:  # keep the worker alive whatever happens
        log.exception("project %s failed", pid)
        _update(pid, status="failed", error=_msg(e))


def _run_render(pid: str, clip_ids: list[int] | None, overrides: dict[str, str]) -> None:
    pdir = project_dir(pid)
    cfg = settings_store.get_all()
    try:
        _check_cancel(pid)
        _update(pid, status="running", stage="render")
        proj = db.one("SELECT * FROM projects WHERE id = ?", (pid,))
        tpath = pdir / "transcript.json"
        if not proj or not proj["video_file"] or not tpath.exists():
            raise RuntimeError("This project has no video or transcript to render from.")
        transcript = json.loads(tpath.read_text(encoding="utf-8"))
        rows = db.rows('SELECT id, start, "end", title, file, opts, edits FROM clips WHERE project_id = ? ORDER BY id', (pid,))
        if clip_ids:
            rows = [r for r in rows if r["id"] in set(clip_ids)]
        if not rows:
            raise RuntimeError("No clips to render.")
        errors = _render_clips(pid, rows, cfg, overrides, pdir, pdir / proj["video_file"], transcript, 0.0, 1.0)
        if len(errors) == len(rows):
            _update(pid, status="failed", error=f"Render failed: {errors[0]}")
        else:
            note = f"{len(errors)} of {len(rows)} clips failed to render; see the red cards." if errors else None
            _update(pid, status="done", stage="done", progress=1.0, error=note)
    except Cancelled:
        _update(pid, status="done", stage="done", progress=1.0, error="Re-render cancelled; clips already finished were kept.")
    except Exception as e:
        log.exception("re-render of project %s failed", pid)
        _update(pid, status="failed", error=_msg(e))
