"""Regression tests for problems found in the code review. Run:  python -m pytest tests -q"""
import json
import shutil
import subprocess
import time
import types

import pytest

from app import db, jobs
from app.pipeline import faces, render, transcribe

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def ff(*args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


# ------------------------------------------------------------------ transcription fallback

class _Seg:
    def __init__(self, a, b, text):
        self.start, self.end, self.text = a, b, text
        self.words = [types.SimpleNamespace(start=a, end=b, word=text)]


class _Model:
    """Stands in for faster-whisper. The GPU version fails on first use, as a missing cublas DLL does."""
    def __init__(self, device):
        self.device = device

    def transcribe(self, *a, **k):
        info = types.SimpleNamespace(duration=10.0, language="en")

        def gen():
            if self.device == "cuda":
                raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
            yield _Seg(0, 5, "hello")
            yield _Seg(5, 10, "world")
        return gen(), info


@pytest.fixture
def fake_whisper(monkeypatch, tmp_path):
    monkeypatch.setattr(transcribe, "extract_audio", lambda v, w: w.write_bytes(b"x"))
    monkeypatch.setattr(transcribe, "_make_model", lambda size, dev: _Model(dev))
    return tmp_path


def test_gpu_failure_during_first_use_falls_back_to_cpu_and_says_why(fake_whisper):
    progress = []
    out = transcribe.transcribe(fake_whisper / "v.mp4", fake_whisper / "t.json", "small", "auto", progress.append)
    assert out["device"] == "cpu" and "cublas" in out["gpu_error"]
    assert [s["text"] for s in out["segments"]] == ["hello", "world"]
    assert not (fake_whisper / "t.wav").exists()
    assert json.loads((fake_whisper / "t.json").read_text())["device"] == "cpu"


def test_explicit_gpu_choice_gives_an_actionable_error(fake_whisper):
    with pytest.raises(RuntimeError, match="Set Device to 'cpu'"):
        transcribe.transcribe(fake_whisper / "v.mp4", fake_whisper / "t.json", "small", "cuda", lambda f: None)


def test_cpu_choice_never_touches_the_gpu(fake_whisper):
    out = transcribe.transcribe(fake_whisper / "v.mp4", fake_whisper / "t.json", "small", "cpu", lambda f: None)
    assert out["device"] == "cpu" and "gpu_error" not in out


@needs_ffmpeg
def test_video_without_audio_gets_a_clear_message(tmp_path):
    ff("-f", "lavfi", "-i", "testsrc=size=160x90:rate=10", "-t", "1", "-c:v", "libx264", str(tmp_path / "v.mp4"))
    with pytest.raises(RuntimeError, match="no audio track"):
        transcribe.extract_audio(tmp_path / "v.mp4", tmp_path / "a.wav")


# ------------------------------------------------------------------ unreadable input

@needs_ffmpeg
def test_audio_only_and_garbage_files_are_rejected_clearly(tmp_path):
    ff("-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(tmp_path / "a.wav"))
    with pytest.raises(RuntimeError, match="no video track"):
        render.probe(tmp_path / "a.wav")
    (tmp_path / "junk.mp4").write_bytes(b"this is not a video" * 50)
    with pytest.raises(RuntimeError, match="could not read"):
        render.probe(tmp_path / "junk.mp4")


# ------------------------------------------------------------------ interrupted jobs and the face cache

def test_jobs_cut_off_by_closing_the_app_are_unlocked(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init_db()
    for pid, st in (("a", "running"), ("b", "queued"), ("c", "done"), ("d", "failed")):
        db.execute("INSERT INTO projects(id, title, status) VALUES(?,?,?)", (pid, pid, st))
    assert jobs.recover_interrupted() == 2
    status = {r["id"]: r["status"] for r in db.rows("SELECT id, status FROM projects")}
    assert status == {"a": "failed", "b": "failed", "c": "done", "d": "failed"}
    assert "closed" in db.one("SELECT error FROM projects WHERE id='a'")["error"]
    assert db.one("SELECT error FROM projects WHERE id='c'")["error"] is None
    assert jobs.recover_interrupted() == 0


@needs_ffmpeg
def test_face_scan_is_cached_and_survives_workdir_cleanup(tmp_path, monkeypatch):
    pytest.importorskip("skimage")
    import cv2
    import numpy as np
    from skimage import data

    face = cv2.resize(cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR), (300, 300))
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "640x360",
                            "-r", "10", "-i", "-", "-f", "lavfi", "-i", "sine=frequency=300", "-t", "3",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                            str(tmp_path / "src.mp4")], stdin=subprocess.PIPE)
    for _ in range(30):
        canvas = np.full((360, 640, 3), 90, np.uint8)
        canvas[30:330, 170:470] = face
        enc.stdin.write(canvas.tobytes())
    enc.stdin.close()
    enc.wait()

    scans = []
    real = faces.scan
    monkeypatch.setattr(faces, "scan", lambda *a, **k: scans.append(1) or real(*a, **k))
    tr = {"segments": [{"start": 0, "end": 3, "text": "a b", "words": [
        {"start": 0.2, "end": 0.6, "word": "hello"}, {"start": 0.8, "end": 1.4, "word": "world"}]}]}
    cfg = {"use_nvenc": "off", "caption_style": "classic"}
    cache = tmp_path / "cache"
    for i, style in enumerate(("classic", "pop")):                    # a restyle must not rescan
        info = render.render_clip(tmp_path / "src.mp4", tmp_path / f"o{i}.mp4", {"start": 0.0, "end": 2.8, "title": "t"},
                                  tr, {**cfg, "caption_style": style}, tmp_path / "work" / f"c{i}",
                                  cache_dir=cache)
        assert info["layout"] == "fill" and (tmp_path / f"o{i}.mp4").exists()
        shutil.rmtree(tmp_path / "work", ignore_errors=True)         # what the job runner does afterwards
    assert len(scans) == 1 and list(cache.glob("faces_*.json"))

    time.sleep(0.01)
    (tmp_path / "src.mp4").write_bytes((tmp_path / "src.mp4").read_bytes() + b"\0")   # source changed on disk
    render.render_clip(tmp_path / "src.mp4", tmp_path / "o3.mp4", {"start": 0.0, "end": 2.8, "title": "t"}, tr, cfg,
                       tmp_path / "work" / "c3", cache_dir=cache)
    assert len(scans) == 2                                            # a changed source invalidates the cache


# ------------------------------------------------------------------ the local API

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient
    mp = pytest.MonkeyPatch()
    mp.setattr(db, "DB_PATH", tmp_path_factory.mktemp("db") / "api.db")
    from app.main import app
    db.init_db()
    yield TestClient(app)
    mp.undo()


def test_only_this_computer_may_use_the_api(client):
    from fastapi.testclient import TestClient
    assert client.get("/api/settings").status_code == 200
    assert TestClient(client.app, base_url="http://localhost:8000").get("/api/settings").status_code == 200
    assert TestClient(client.app, base_url="http://[::1]:8000").get("/api/settings").status_code == 200
    assert TestClient(client.app, base_url="http://evil.example").get("/api/settings").status_code == 403   # DNS rebinding


def test_other_websites_cannot_change_settings(client):
    body = {"caption_style": "classic"}
    assert client.put("/api/settings", json=body, headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.put("/api/settings", json=body, headers={"Origin": "null"}).status_code == 403
    assert client.put("/api/settings", json=body, headers={"Origin": "http://127.0.0.1:8000"}).status_code == 200
    assert client.put("/api/settings", json=body).status_code == 200


def test_settings_reject_values_that_would_break_a_job_later(client):
    for bad in ({"whisper_model": "huge"}, {"whisper_device": "tpu"}, {"max_clips": "2.5"}, {"max_clips": "0"},
                {"max_clips": "99"}, {"min_clip_seconds": "90", "max_clip_seconds": "20"}, {"silence_gap": "0"}):
        assert client.put("/api/settings", json=bad).status_code == 400, bad
    assert client.put("/api/settings", json={"whisper_model": "medium", "max_clips": "5"}).status_code == 200
    client.put("/api/settings", json={"whisper_model": "small", "max_clips": "8"})


def test_a_running_project_cannot_be_deleted_but_an_unlocked_one_can(client):
    db.execute("INSERT INTO projects(id, title, status) VALUES('busy1', 'x', 'running')")
    assert client.delete("/api/projects/busy1").status_code == 409
    jobs.recover_interrupted()
    assert client.delete("/api/projects/busy1").status_code == 200
    assert client.get("/api/projects/busy1").status_code == 404


@needs_ffmpeg
def test_audio_only_upload_fails_fast_with_a_readable_reason(client, tmp_path):
    ff("-f", "lavfi", "-i", "sine=frequency=440:duration=2", str(tmp_path / "talk.wav"))
    pid = client.post("/api/projects", files={"file": ("talk.wav", (tmp_path / "talk.wav").read_bytes(), "audio/wav")}).json()["id"]
    for _ in range(60):
        p = client.get(f"/api/projects/{pid}").json()
        if p["status"] in ("failed", "done"):
            break
        time.sleep(0.25)
    assert p["status"] == "failed" and p["error"].startswith("This file has no video track")
    assert client.delete(f"/api/projects/{pid}").status_code == 200
