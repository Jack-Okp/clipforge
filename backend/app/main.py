"""ClipForge local API. Run with:  python -m uvicorn app.main:app --port 8000"""
import json
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.background import BackgroundTask

from . import config, db, jobs, settings_store
from .config import BASE_DIR, BRAND_DIR, DATA_DIR, FRONTEND_DIR, KITS_DIR, PROJECTS_DIR
from .pipeline import broll, captions, timeline
from .providers import PROVIDERS, LLMError, get_provider
from .providers.ollama import list_models

load_dotenv(BASE_DIR / ".env")
db.init_db()
config.setup_logging()
jobs.recover_interrupted()

app = FastAPI(title="ClipForge")
app.mount("/media", StaticFiles(directory=PROJECTS_DIR), name="media")

# This API holds your API keys and can start downloads, so only the page served from this very
# machine may use it. Without this, any website open in another tab could call it.
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "testserver"}


def _hostname(value: str) -> str:
    value = value.strip()
    if value.startswith("["):                      # IPv6 literal such as [::1]:8000
        return value[1:value.find("]")] if "]" in value else value
    return value.split(":")[0].lower()


@app.middleware("http")
async def local_only(request: Request, call_next):
    # A wrong Host header means DNS rebinding; a foreign Origin on a write means another site is calling us.
    if _hostname(request.headers.get("host", "")) not in LOCAL_HOSTS:
        return JSONResponse({"detail": "ClipForge only answers on 127.0.0.1 / localhost."}, status_code=403)
    origin = request.headers.get("origin")
    if request.method not in ("GET", "HEAD", "OPTIONS") and origin and origin != "null":
        if _hostname(urlparse(origin).netloc) not in LOCAL_HOSTS:
            return JSONResponse({"detail": "Cross-site request blocked."}, status_code=403)
    if request.method not in ("GET", "HEAD", "OPTIONS") and origin == "null":
        return JSONResponse({"detail": "Cross-site request blocked."}, status_code=403)
    return await call_next(request)


# ---------- settings & model switching ----------

@app.get("/api/settings")
def get_settings():
    return {
        "settings": settings_store.public_view(),
        "providers": [{"id": k, "label": v.label} for k, v in PROVIDERS.items()],
    }


@app.put("/api/settings")
def put_settings(values: dict[str, str]):
    clean = {}
    for k, v in values.items():
        if k == "watermark_path":
            continue  # only the logo upload may set this
        if k in settings_store.SECRET_KEYS and not v.strip():
            continue  # blank secret field = keep the stored key
        clean[k] = v.strip()
    if clean.get("llm_provider") and clean["llm_provider"] not in PROVIDERS:
        raise HTTPException(400, "Unknown provider")
    bad = settings_store.validate({k: v for k, v in clean.items() if k not in settings_store.SECRET_KEYS})
    if bad:
        raise HTTPException(400, bad)
    settings_store.set_many(clean)
    return {"ok": True}


# ---------- brand logo (watermark) ----------

@app.post("/api/brand/logo")
async def upload_logo(file: UploadFile = File(...)):
    import cv2
    import numpy as np
    ext = Path(file.filename or "").suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
        raise HTTPException(400, "Logo must be a PNG, JPG or WEBP image")
    data = await file.read()
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(400, "Logo is larger than 5 MB")
    if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED) is None:
        raise HTTPException(400, "That file is not a readable image")
    for old in BRAND_DIR.glob("logo.*"):
        old.unlink()
    path = BRAND_DIR / f"logo{ext}"
    path.write_bytes(data)
    settings_store.set_many({"watermark_path": str(path)})
    return {"ok": True}


@app.delete("/api/brand/logo")
def delete_logo():
    for old in BRAND_DIR.glob("logo.*"):
        old.unlink()
    settings_store.set_many({"watermark_path": "", "watermark": "off"})
    return {"ok": True}


@app.get("/api/brand/logo")
def get_logo():
    path = settings_store.get_all()["watermark_path"]
    if not path or not Path(path).exists():
        raise HTTPException(404, "No logo uploaded")
    return FileResponse(path)


@app.delete("/api/settings/key/{provider}")
def clear_key(provider: str):
    field = {"claude": "anthropic_api_key", "gemini": "gemini_api_key"}.get(provider)
    if not field:
        raise HTTPException(400, "That provider has no API key")
    settings_store.set_many({field: ""})
    return {"ok": True}


class TestRequest(BaseModel):
    provider: str
    values: dict[str, str] = {}  # unsaved form values, so you can test before saving


@app.post("/api/settings/test")
def test_provider(req: TestRequest):
    cfg = settings_store.get_all()
    cfg.update({k: v.strip() for k, v in req.values.items() if v.strip()})
    try:
        reply = get_provider(cfg, req.provider).test()
        return {"ok": True, "reply": reply}
    except LLMError as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/ollama/models")
def ollama_models(base_url: str = ""):
    cfg = settings_store.get_all()
    try:
        return {"ok": True, "models": list_models(base_url or cfg["ollama_base_url"])}
    except LLMError as e:
        return {"ok": False, "error": str(e), "models": []}


# ---------- projects ----------

@app.get("/api/projects")
def list_projects():
    projects = db.rows("SELECT * FROM projects ORDER BY created_at DESC")
    order = {p["id"]: i for i, p in enumerate(_queue_rows())}
    for p in projects:
        p["queue_position"] = order.get(p["id"])       # 0 = working now, 1 = next, ...
    return projects


def _queue_rows() -> list[dict]:
    rows = db.rows("SELECT id, title, status, stage, progress, queued_at FROM projects "
                   "WHERE status IN ('running', 'queued')")
    rows.sort(key=lambda r: (r["status"] != "running", r["queued_at"] or 0))
    return rows


@app.get("/api/queue")
def queue():
    return _queue_rows()


@app.post("/api/queue/cancel-all")
def cancel_all():
    rows = _queue_rows()
    for r in rows:
        jobs.cancel(r["id"])
    return {"cancelled": len(rows)}


@app.post("/api/queue/retry-failed")
def retry_failed():
    n = 0
    for p in db.rows("SELECT id FROM projects WHERE status = 'failed' ORDER BY created_at"):
        stage = "highlights" if (PROJECTS_DIR / p["id"] / "transcript.json").exists() else "ingest"
        jobs.enqueue(p["id"], None, from_stage=stage)
        n += 1
    return {"retried": n}


@app.post("/api/projects")
def create_project(url: str = Form(""), focus: str = Form(""), file: UploadFile | None = File(None)):
    # A plain def (not async) so copying a multi-gigabyte upload does not freeze the whole server.
    pid = uuid.uuid4().hex[:10]
    pdir = jobs.project_dir(pid)
    url = url.strip()
    if file and file.filename:
        safe = "source" + Path(file.filename).suffix.lower()
        with open(pdir / safe, "wb") as out:
            shutil.copyfileobj(file.file, out)
        db.execute("INSERT INTO projects(id, title, source, video_file) VALUES(?,?,?,?)",
                   (pid, Path(file.filename).stem, file.filename, safe))
    elif url.startswith(("http://", "https://")):
        db.execute("INSERT INTO projects(id, title, source) VALUES(?,?,?)", (pid, url, url))
    else:
        shutil.rmtree(pdir, ignore_errors=True)
        raise HTTPException(400, "Provide a video file or a http(s) link")
    jobs.enqueue(pid, focus=focus)
    return {"id": pid}


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    p = db.one("SELECT * FROM projects WHERE id = ?", (pid,))
    if not p:
        raise HTTPException(404, "Project not found")
    p["clips"] = db.rows(f"SELECT {CLIP_COLS} FROM clips WHERE project_id = ? ORDER BY score DESC", (pid,))
    return p


# every clip column except the (large) caption word list
CLIP_COLS = ('id, project_id, start, "end", title, score, reason, file, provider, layout, encoder, out_seconds, '
             "removed_seconds, rendered_at, opts, error, edits")


@app.get("/api/projects/{pid}/transcript")
def get_transcript(pid: str):
    t = PROJECTS_DIR / pid / "transcript.json"
    if not t.exists():
        raise HTTPException(404, "No transcript yet")
    return json.loads(t.read_text(encoding="utf-8"))


class RerunRequest(BaseModel):
    provider: str | None = None  # re-pick clips with a different model, no re-transcribing
    focus: str | None = None     # e.g. "moments about pricing"; "" clears it, None keeps the saved one


@app.post("/api/projects/{pid}/rerun")
def rerun(pid: str, req: RerunRequest):
    p = db.one("SELECT id, status FROM projects WHERE id = ?", (pid,))
    if not p:
        raise HTTPException(404, "Project not found")
    if p["status"] in ("running", "queued"):
        raise HTTPException(409, "This project is busy. Wait for it to finish or cancel it first.")
    if req.provider and req.provider not in PROVIDERS:
        raise HTTPException(400, "Unknown provider")
    stage = "highlights" if (PROJECTS_DIR / pid / "transcript.json").exists() else "ingest"
    jobs.enqueue(pid, req.provider, from_stage=stage, focus=req.focus)
    return {"ok": True}


class RenderRequest(BaseModel):
    opts: dict[str, str] = {}          # style overrides for this render only (aspect, captions, ...)
    clip_ids: list[int] | None = None  # None = every clip in the project
    kit_id: int | None = None          # apply a saved brand kit (its style and logo) to this render


@app.post("/api/projects/{pid}/render")
def render_project(pid: str, req: RenderRequest):
    """Restyle clips (new captions, aspect, cleanup...) without re-transcribing or re-asking the AI."""
    p = db.one("SELECT id, status FROM projects WHERE id = ?", (pid,))
    if not p:
        raise HTTPException(404, "Project not found")
    if p["status"] in ("running", "queued"):
        raise HTTPException(409, "This project is busy. Wait for it to finish or cancel it first.")
    opts = {k: v.strip() for k, v in req.opts.items() if k in settings_store.STYLE_KEYS}
    bad = settings_store.validate(opts)
    if bad:
        raise HTTPException(400, bad)
    if req.kit_id is not None:
        kit = _kit(req.kit_id)
        opts = {**kit["opts"], **opts}
        if kit["logo_path"]:
            opts["watermark_path"] = kit["logo_path"]
    jobs.enqueue_render(pid, req.clip_ids, opts)
    return {"ok": True}


@app.post("/api/projects/{pid}/cancel")
def cancel(pid: str):
    jobs.cancel(pid)
    return {"ok": True}


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    p = db.one("SELECT status FROM projects WHERE id = ?", (pid,))
    if p and p["status"] in ("running", "queued"):
        raise HTTPException(409, "This project is still running. Cancel it first, then delete it.")
    db.execute("DELETE FROM clips WHERE project_id = ?", (pid,))
    db.execute("DELETE FROM projects WHERE id = ?", (pid,))
    shutil.rmtree(PROJECTS_DIR / pid, ignore_errors=True)
    return {"ok": True}


# ---------- brand kits ----------

def _kit(kit_id: int) -> dict:
    r = db.one("SELECT * FROM kits WHERE id = ?", (kit_id,))
    if not r:
        raise HTTPException(404, "Kit not found")
    logo = KITS_DIR / r["logo"] if r["logo"] else None
    return {"id": r["id"], "name": r["name"], "opts": json.loads(r["opts"] or "{}"),
            "logo_path": str(logo) if logo and logo.exists() else None}


@app.get("/api/kits")
def list_kits():
    out = []
    for r in db.rows("SELECT id FROM kits ORDER BY name COLLATE NOCASE"):
        k = _kit(r["id"])
        out.append({"id": k["id"], "name": k["name"], "opts": k["opts"], "has_logo": bool(k["logo_path"])})
    return out


class KitRequest(BaseModel):
    name: str
    opts: dict[str, str] | None = None   # None = the style currently saved in Settings


@app.post("/api/kits")
def create_kit(req: KitRequest):
    name = req.name.strip()[:60]
    if not name:
        raise HTTPException(400, "Give the kit a name")
    cfg = settings_store.get_all()
    opts = {k: cfg[k] for k in settings_store.STYLE_KEYS}
    if req.opts:
        opts.update({k: v.strip() for k, v in req.opts.items() if k in settings_store.STYLE_KEYS})
    bad = settings_store.validate(opts)
    if bad:
        raise HTTPException(400, bad)
    kid = db.execute("INSERT INTO kits(name, opts) VALUES(?, ?)", (name, json.dumps(opts)))
    logo = Path(cfg["watermark_path"]) if cfg["watermark_path"] else None
    if logo and logo.exists():
        fname = f"kit_{kid}{logo.suffix}"
        shutil.copyfile(logo, KITS_DIR / fname)
        db.execute("UPDATE kits SET logo = ? WHERE id = ?", (fname, kid))
    return {"id": kid}


@app.post("/api/kits/{kit_id}/apply")
def apply_kit(kit_id: int):
    """Make the kit the default style for new projects (and put its logo in place)."""
    kit = _kit(kit_id)
    values = dict(kit["opts"])
    if kit["logo_path"]:
        for old in BRAND_DIR.glob("logo.*"):
            old.unlink()
        dst = BRAND_DIR / f"logo{Path(kit['logo_path']).suffix}"
        shutil.copyfile(kit["logo_path"], dst)
        values["watermark_path"] = str(dst)
    settings_store.set_many(values)
    return {"ok": True}


@app.get("/api/kits/{kit_id}/logo")
def kit_logo(kit_id: int):
    path = _kit(kit_id)["logo_path"]
    if not path:
        raise HTTPException(404, "This kit has no logo")
    return FileResponse(path)


@app.delete("/api/kits/{kit_id}")
def delete_kit(kit_id: int):
    kit = _kit(kit_id)
    if kit["logo_path"]:
        Path(kit["logo_path"]).unlink(missing_ok=True)
    db.execute("DELETE FROM kits WHERE id = ?", (kit_id,))
    return {"ok": True}


# ---------- clip editor ----------

def _clip(clip_id: int) -> tuple[dict, dict]:
    c = db.one(f"SELECT {CLIP_COLS} FROM clips WHERE id = ?", (clip_id,))
    if not c:
        raise HTTPException(404, "Clip not found")
    p = db.one("SELECT id, title, video_file, duration, status FROM projects WHERE id = ?", (c["project_id"],))
    return c, p


def _loads(text, default):
    try:
        return json.loads(text) if text else default
    except ValueError:
        return default


@app.get("/api/clips/{clip_id}")
def get_clip(clip_id: int):
    c, p = _clip(clip_id)
    cfg = settings_store.get_all()
    style = {k: cfg[k] for k in settings_store.STYLE_KEYS}
    style.update({k: v for k, v in _loads(c["opts"], {}).items() if k in settings_store.STYLE_KEYS})
    c["edits"] = _loads(c["edits"], {})
    c["opts"] = _loads(c["opts"], {})
    return {"clip": c, "project": p, "style": style, "broll_enabled": bool(cfg["broll_folder"])}


@app.get("/api/clips/{clip_id}/words")
def clip_words(clip_id: int, pad: float = 20.0):
    """Transcript words around the clip, so the editor can move the start and end into neighbouring speech."""
    c, p = _clip(clip_id)
    t = PROJECTS_DIR / p["id"] / "transcript.json"
    if not t.exists():
        raise HTTPException(404, "No transcript for this project")
    pad = min(max(pad, 0.0), 120.0)
    lo = max(c["start"] - pad, 0.0)
    hi = min(c["end"] + pad, p["duration"] or c["end"] + pad)
    words = timeline.clip_words(json.loads(t.read_text(encoding="utf-8")), lo, hi)
    return {"words": [{"s": round(w["start"], 2), "e": round(w["end"], 2), "w": w["word"], "k": w["key"]} for w in words]}


MAX_LEN, MIN_LEN = 180.0, 3.0


def _clean_edits(raw, cfg: dict) -> dict | None:
    """Accept only the editor's known fields with sane limits, so a bad request cannot break a render later."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise HTTPException(400, "edits must be an object")
    out: dict = {}
    words = raw.get("words") or {}
    if not isinstance(words, dict) or len(words) > 3000 or not all(isinstance(k, str) and isinstance(v, str) for k, v in words.items()):
        raise HTTPException(400, "edits.words must map word ids to text")
    if words:
        out["words"] = {k: v.strip()[:60] for k, v in words.items()}
    cuts = raw.get("cuts") or []
    if not isinstance(cuts, list) or len(cuts) > 3000 or not all(isinstance(k, str) for k in cuts):
        raise HTTPException(400, "edits.cuts must be a list of word ids")
    if cuts:
        out["cuts"] = sorted(set(cuts))
    crop = raw.get("crop")
    if crop:
        try:
            cx, cy, zoom = float(crop["cx"]), float(crop["cy"]), float(crop.get("zoom", 1))
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, "edits.crop needs cx, cy and zoom")
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 1 <= zoom <= 3):
            raise HTTPException(400, "Crop position must be inside the picture and zoom between 1 and 3")
        out["crop"] = {"cx": round(cx, 4), "cy": round(cy, 4), "zoom": round(zoom, 3)}
    items = raw.get("broll") or []
    if not isinstance(items, list) or len(items) > 12:
        raise HTTPException(400, "At most 12 B-roll cut-aways per clip")
    folder = cfg.get("broll_folder", "")
    clean_items = []
    for it in items:
        try:
            path, at, dur = str(it["path"]), float(it["at"]), float(it["dur"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, "Each B-roll item needs path, at and dur")
        if not folder or not broll.is_inside(path, folder) or not Path(path).is_file():
            raise HTTPException(400, "B-roll files must come from your B-roll folder (set it in Settings)")
        if at < 0 or not (0.5 <= dur <= 15):
            raise HTTPException(400, "A B-roll cut-away must last from 0.5 to 15 seconds")
        clean_items.append({"path": path, "at": round(at, 2), "dur": round(dur, 2)})
    if clean_items:
        out["broll"] = clean_items
    return out or None


class ClipUpdate(BaseModel):
    title: str | None = None
    start: float | None = None
    end: float | None = None
    edits: dict | None = None        # replaces the clip's saved edits (send {} to clear)
    opts: dict[str, str] | None = None   # style for this clip only


@app.put("/api/clips/{clip_id}")
def update_clip(clip_id: int, req: ClipUpdate):
    """Save the editor's changes and re-render just this clip."""
    c, p = _clip(clip_id)
    if p["status"] in ("running", "queued"):
        raise HTTPException(409, "This project is busy. Wait for it to finish or cancel it first.")
    cfg = settings_store.get_all()
    start = c["start"] if req.start is None else float(req.start)
    end = c["end"] if req.end is None else float(req.end)
    if start < 0 or (p["duration"] and end > p["duration"] + 0.5):
        raise HTTPException(400, "The clip must stay inside the video")
    if not (MIN_LEN <= end - start <= MAX_LEN):
        raise HTTPException(400, f"A clip must be between {MIN_LEN:.0f} and {MAX_LEN:.0f} seconds long")
    title = c["title"] if req.title is None else req.title.strip()[:120]
    fields: dict = {"start": round(start, 2), "end": round(end, 2), "title": title}
    if req.edits is not None:
        cleaned = _clean_edits(req.edits, cfg)
        fields["edits"] = json.dumps(cleaned) if cleaned else None
    if req.opts is not None:
        opts = {k: v.strip() for k, v in req.opts.items() if k in settings_store.STYLE_KEYS}
        bad = settings_store.validate(opts)
        if bad:
            raise HTTPException(400, bad)
        fields["opts"] = json.dumps({**_loads(c["opts"], {}), **opts})
    cols = ", ".join(f'"{k}" = ?' for k in fields)
    db.execute(f"UPDATE clips SET {cols} WHERE id = ?", (*fields.values(), clip_id))
    jobs.enqueue_render(p["id"], [clip_id], {})
    return {"ok": True}


def _clip_srt(c: dict, p: dict) -> str:
    row = db.one("SELECT words FROM clips WHERE id = ?", (c["id"],))
    words = _loads(row["words"], None) if row else None
    if words is None:                      # clip rendered before Phase 3: use the transcript instead
        t = PROJECTS_DIR / p["id"] / "transcript.json"
        if not t.exists():
            raise HTTPException(404, "No captions available for this clip yet")
        words = [{**w, "start": w["start"] - c["start"], "end": w["end"] - c["start"]}
                 for w in timeline.clip_words(json.loads(t.read_text(encoding="utf-8")), c["start"], c["end"])]
    return captions.build_srt(words)


@app.get("/api/clips/{clip_id}/srt")
def clip_srt(clip_id: int):
    c, p = _clip(clip_id)
    name = Path(c["file"] or f"clip_{clip_id}.mp4").stem + ".srt"
    return PlainTextResponse(_clip_srt(c, p), headers={"Content-Disposition": f'attachment; filename="{name}"'})


# ---------- exporting ----------

def _safe_name(text: str) -> str:
    import re
    return re.sub(r"[^\w\- ]+", "", text or "").strip()[:60] or "clip"


def _finished(pid: str) -> list[tuple[dict, Path]]:
    p = db.one("SELECT id FROM projects WHERE id = ?", (pid,))
    if not p:
        raise HTTPException(404, "Project not found")
    out = []
    for c in db.rows(f"SELECT {CLIP_COLS} FROM clips WHERE project_id = ? ORDER BY score DESC", (pid,)):
        f = PROJECTS_DIR / pid / "clips" / (c["file"] or "")
        if c["file"] and f.exists() and not c["error"]:
            out.append((c, f))
    if not out:
        raise HTTPException(404, "No finished clips to export yet")
    return out


@app.get("/api/projects/{pid}/zip")
def project_zip(pid: str):
    clips = _finished(pid)
    proj = db.one("SELECT * FROM projects WHERE id = ?", (pid,))
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp.close()
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_STORED) as z:       # mp4 is already compressed
        for n, (c, f) in enumerate(clips, 1):
            base = f"{n:02d}_{_safe_name(c['title'])}"
            z.write(f, base + ".mp4")
            z.writestr(base + ".srt", _clip_srt(c, proj))
    return FileResponse(tmp.name, media_type="application/zip", filename=f"{_safe_name(proj['title'])}_clips.zip",
                        background=BackgroundTask(lambda: Path(tmp.name).unlink(missing_ok=True)))


class FolderRequest(BaseModel):
    folder: str


@app.post("/api/projects/{pid}/export-folder")
def export_folder(pid: str, req: FolderRequest):
    """Copy the finished clips (and subtitle files) into a folder on this computer."""
    clips = _finished(pid)
    proj = db.one("SELECT * FROM projects WHERE id = ?", (pid,))
    if not req.folder.strip():
        raise HTTPException(400, "Choose a folder")
    dest = Path(req.folder.strip()).expanduser()
    try:
        dest.mkdir(parents=True, exist_ok=True)
        for n, (c, f) in enumerate(clips, 1):
            base = f"{n:02d}_{_safe_name(c['title'])}"
            shutil.copyfile(f, dest / (base + ".mp4"))
            (dest / (base + ".srt")).write_text(_clip_srt(c, proj), encoding="utf-8")
    except OSError as e:
        raise HTTPException(400, f"Could not write to that folder: {e.strerror or e}")
    return {"ok": True, "count": len(clips), "folder": str(dest)}


# ---------- B-roll from your own footage ----------

def _broll_root() -> str:
    folder = settings_store.get_all()["broll_folder"]
    if not folder or not Path(folder).is_dir():
        raise HTTPException(400, "Set a B-roll folder in Settings first")
    return folder


@app.get("/api/broll/list")
def broll_list():
    items = broll.scan(_broll_root())
    return [{k: v for k, v in i.items() if k != "tokens"} for i in items]


@app.get("/api/broll/suggest")
def broll_suggest(clip_id: int):
    folder = _broll_root()
    c, p = _clip(clip_id)
    t = PROJECTS_DIR / p["id"] / "transcript.json"
    if not t.exists():
        raise HTTPException(404, "No transcript for this project")
    words = timeline.clip_words(json.loads(t.read_text(encoding="utf-8")), c["start"], c["end"])
    hits = broll.suggest(broll.scan(folder), words)
    for h in hits:
        h["at"] = round(max(h["at"] - c["start"], 0.0), 2)           # relative to the clip start
    return {"suggestions": hits}


@app.get("/api/broll/thumb")
def broll_thumb(path: str):
    folder = _broll_root()
    if not broll.is_inside(path, folder):
        raise HTTPException(403, "Not inside your B-roll folder")
    th = broll.thumb(path)
    if not th:
        raise HTTPException(404, "No preview available")
    return FileResponse(th, media_type="image/jpeg")


# ---------- system check ----------

@app.get("/api/health")
def health():
    import importlib.util
    import subprocess
    from .pipeline import export
    checks = []

    def add(cid, label, ok, detail, hint=""):
        checks.append({"id": cid, "label": label, "ok": bool(ok), "detail": detail, "hint": "" if ok else hint})

    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        ver = ""
        if path:
            try:
                ver = subprocess.run([tool, "-version"], capture_output=True, text=True, timeout=10).stdout.splitlines()[0][:70]
            except Exception:
                path = None
        add(tool, tool.upper(), path, ver or "not found", "Install with: winget install Gyan.FFmpeg, then reopen the window.")
    try:
        nv = export.nvenc_available()
    except Exception:
        nv = False
    add("nvenc", "GPU video encoding (NVENC)", nv, "available" if nv else "not available - CPU encoding is used",
        "Optional. Update your NVIDIA driver to enable it.")
    cuda = 0
    try:
        import ctranslate2
        cuda = ctranslate2.get_cuda_device_count()
    except Exception:
        pass
    add("cuda", "GPU for transcription (CUDA)", cuda, f"{cuda} device(s)" if cuda else "not available - transcription uses the CPU",
        "Optional but much faster. Re-run setup.bat and make sure your NVIDIA driver is current.")
    for mod, label in (("faster_whisper", "faster-whisper"), ("cv2", "OpenCV"), ("yt_dlp", "yt-dlp")):
        add(mod, label, importlib.util.find_spec(mod), "installed" if importlib.util.find_spec(mod) else "missing",
            "Run setup.bat again.")
    free = shutil.disk_usage(DATA_DIR).free / 1e9
    add("disk", "Free disk space", free > 5, f"{free:.0f} GB free", "Clips and downloads need several GB.")
    cfg = settings_store.get_all()
    for prov, field in (("claude", "anthropic_api_key"), ("gemini", "gemini_api_key")):
        add(f"key_{prov}", f"{prov.title()} API key", cfg[field], "saved" if cfg[field] else "not set",
            "Only needed if you use this provider. Add it in Settings.")
    try:
        models = list_models(cfg["ollama_base_url"])
        add("ollama", "Ollama", True, f"running, {len(models)} model(s)")
    except LLMError as e:
        add("ollama", "Ollama", False, "not reachable", "Only needed if you use Ollama. Start the Ollama app.")
    bf = cfg["broll_folder"]
    add("broll", "B-roll folder", (not bf) or Path(bf).is_dir(), bf or "not set (optional)", "The saved folder no longer exists.")
    return {"checks": checks, "ok": all(c["ok"] for c in checks if c["id"] in ("ffmpeg", "ffprobe", "faster_whisper", "cv2"))}


# ---------- frontend ----------

@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")
