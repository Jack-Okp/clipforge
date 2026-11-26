"""Phase 3 tests. Run:  python -m pytest tests -q"""
import pytest

from app.pipeline import highlights
from app.providers.base import LLMError


class Flaky:
    """Provider stub that plays back a script of replies / errors."""
    name = "stub"

    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def generate(self, *a, **k):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def waits(monkeypatch):
    got = []
    monkeypatch.setattr(highlights, "_sleep", got.append)
    return got


def test_transient_errors_back_off_then_succeed(waits):
    p = Flaky([LLMError("busy", transient=True), LLMError("busy", transient=True), '{"clips": []}'])
    assert highlights._ask(p, "x") == {"clips": []}
    assert p.calls == 3 and waits == [3, 8]


def test_transient_errors_give_up_after_three_attempts(waits):
    p = Flaky([LLMError("busy", transient=True)] * 5)
    with pytest.raises(LLMError, match="busy"):
        highlights._ask(p, "x")
    assert p.calls == 3


def test_permanent_errors_fail_at_once(waits):
    p = Flaky([LLMError("bad key"), '{"clips": []}'])
    with pytest.raises(LLMError, match="bad key"):
        highlights._ask(p, "x")
    assert p.calls == 1 and waits == []


def test_bad_json_is_retried_once_only(waits):
    p = Flaky(["nonsense", '{"clips": []}'])
    assert highlights._ask(p, "x") == {"clips": []}
    p = Flaky(["nonsense", "still nonsense", '{"clips": []}'])
    with pytest.raises(LLMError, match="valid JSON"):
        highlights._ask(p, "x")
    assert p.calls == 2


def test_focus_text_is_cleaned():
    assert highlights.clean_focus('  moments   about "pricing"\n') == "moments about 'pricing'"
    assert len(highlights.clean_focus("x" * 500)) == 200
    assert highlights.clean_focus(None) == ""


# ------------------------------------------------------------------ timeline, captions, sizes
import shutil
import subprocess

from app.pipeline import captions, render, timeline

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")

TR = {"segments": [{"start": 0, "end": 4, "text": "x", "words": [
    {"start": 0.2, "end": 0.6, "word": "hello"}, {"start": 0.7, "end": 1.1, "word": "big"},
    {"start": 1.2, "end": 1.6, "word": "wide"}, {"start": 1.7, "end": 2.2, "word": "world."},
    {"start": 2.4, "end": 2.9, "word": "again"}]}]}


def test_words_get_stable_keys_that_survive_remapping():
    words = timeline.clip_words(TR, 0.0, 4.0)
    assert [w["key"] for w in words] == ["0.20", "0.70", "1.20", "1.70", "2.40"]
    tl = timeline.compute_timeline(0.0, 4.0, words, 30, False, False, extra_cuts=[(0.7, 1.1)])
    kept = [w["key"] for w in tl.remap_words(words)]
    assert "0.70" not in kept and "1.20" in kept
    assert tl.removed_seconds > 0.3


def test_text_edits_change_or_hide_words():
    words = timeline.clip_words(TR, 0.0, 4.0)
    out = timeline.apply_text_edits(words, {"0.20": "Howdy", "0.70": "", "bogus": "x"})
    assert [w["word"] for w in out] == ["Howdy", "wide", "world.", "again"]
    assert timeline.apply_text_edits(words, None) is words


def test_srt_has_valid_numbering_and_no_overlaps():
    words = timeline.clip_words(TR, 0.0, 4.0)
    srt = captions.build_srt(words, max_words=3)
    blocks = [b.splitlines() for b in srt.strip().split("\n\n")]
    assert [b[0] for b in blocks] == [str(i) for i in range(1, len(blocks) + 1)] and len(blocks) >= 2
    ends = []
    for b in blocks:
        a, z = b[1].split(" --> ")
        assert a < z and (not ends or a >= ends[-1])
        ends.append(z)
    assert captions.srt_time(3661.5) == "01:01:01,500"


def test_output_size_follows_export_width():
    assert render.output_size("9:16", 1920, 1080) == (1080, 1920)
    assert render.output_size("9:16", 1920, 1080, 720) == (720, 1280)
    assert render.output_size("4:5", 1920, 1080, 720) == (720, 900)
    assert render.output_size("original", 3840, 2160, 720) == (1280, 720)
    assert render.output_size("original", 3840, 2160) == (1920, 1080)


def test_export_presets_are_parsed_with_safe_fallbacks():
    assert render.parse_opts({"export_quality": "small", "export_width": "720"})["quality"] == "small"
    o = render.parse_opts({"export_quality": "weird", "export_width": "999"})
    assert o["quality"] == "high" and o["width"] == 1080
    assert render.needs_render({"output_aspect": "original", "caption_style": "none", "remove_fillers": "off",
                                "normalize_audio": "off", "export_quality": "small"})


# ------------------------------------------------------------------ real renders with editor edits

def _ff(*a):
    subprocess.run(["ffmpeg", "-y", "-v", "error", *a], check=True)


def _frame(video, t, tmp):
    import cv2
    png = tmp / f"f_{t}.png"
    _ff("-ss", str(t), "-i", str(video), "-frames:v", "1", str(png))
    return cv2.imread(str(png))


@pytest.fixture(scope="module")
def halves(tmp_path_factory):
    """Left half red, right half blue, with a tone, 4 s."""
    d = tmp_path_factory.mktemp("halves")
    _ff("-f", "lavfi", "-i", "color=c=red:size=320x360:rate=15", "-f", "lavfi", "-i", "color=c=blue:size=320x360:rate=15",
        "-f", "lavfi", "-i", "sine=frequency=300", "-filter_complex", "[0][1]hstack", "-t", "4",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(d / "halves.mp4"))
    return d


BASE = {"use_nvenc": "off", "caption_style": "none", "remove_fillers": "off", "normalize_audio": "off",
        "output_aspect": "9:16", "export_width": "720"}


@needs_ffmpeg
def test_manual_crop_follows_the_box_and_skips_face_scan(halves, monkeypatch):
    from app.pipeline import faces
    monkeypatch.setattr(faces, "load_or_scan", lambda *a, **k: pytest.fail("face scan must not run"))
    clip = {"start": 0.0, "end": 3.0, "title": "t"}
    left = render.render_clip(halves / "halves.mp4", halves / "l.mp4", clip, TR, BASE, halves / "w1",
                              edits={"crop": {"cx": 0.25, "cy": 0.5, "zoom": 1.0}})
    render.render_clip(halves / "halves.mp4", halves / "r.mp4", clip, TR, BASE, halves / "w2",
                       edits={"crop": {"cx": 0.8, "cy": 0.5, "zoom": 1.5}})
    assert left["layout"] == "manual" and (left["width"], left["height"]) == (720, 1280)
    f_l, f_r = _frame(halves / "l.mp4", 1, halves), _frame(halves / "r.mp4", 1, halves)
    assert f_l[640, 360, 2] > 200 and f_l[640, 360, 0] < 60          # red (BGR)
    assert f_r[640, 360, 0] > 200 and f_r[640, 360, 2] < 60          # blue


@needs_ffmpeg
def test_cutting_words_shortens_the_clip_and_returns_final_words(halves):
    clip = {"start": 0.0, "end": 3.0, "title": "t"}
    plain = render.render_clip(halves / "halves.mp4", halves / "p.mp4", clip, TR, BASE, halves / "w3")
    cut = render.render_clip(halves / "halves.mp4", halves / "c.mp4", clip, TR, BASE, halves / "w4",
                             edits={"cuts": ["0.70", "1.20"], "words": {"0.20": "Howdy"}})
    assert cut["out_seconds"] < plain["out_seconds"] - 0.6
    assert [w["word"] for w in cut["words"]] == ["Howdy", "world.", "again"]
    assert all(a["end"] <= b["start"] + 0.06 for a, b in zip(cut["words"], cut["words"][1:]))


@needs_ffmpeg
def test_broll_image_replaces_the_picture_but_not_the_audio(halves):
    import cv2
    import numpy as np
    cv2.imwrite(str(halves / "mag.png"), np.full((200, 300, 3), (255, 0, 255), np.uint8))
    clip = {"start": 0.0, "end": 3.0, "title": "t"}
    info = render.render_clip(halves / "halves.mp4", halves / "b.mp4", clip, TR, BASE, halves / "w5",
                              edits={"crop": {"cx": 0.25, "cy": 0.5, "zoom": 1.0},
                                     "broll": [{"path": str(halves / "mag.png"), "at": 1.0, "dur": 1.0},
                                               {"path": str(halves / "missing.mp4"), "at": 0.1, "dur": 0.5}]})
    inside, outside = _frame(halves / "b.mp4", 1.5, halves), _frame(halves / "b.mp4", 0.5, halves)
    assert inside[640, 360, 0] > 200 and inside[640, 360, 1] < 60 and inside[640, 360, 2] > 200   # magenta
    assert outside[640, 360, 2] > 200 and outside[640, 360, 0] < 60                                # red again
    assert info["out_seconds"] == pytest.approx(3.0, abs=0.1)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type",
                            "-of", "csv=p=0", str(halves / "b.mp4")], capture_output=True, text=True).stdout
    assert "audio" in probe


@needs_ffmpeg
def test_broll_video_file_is_cover_cropped(halves):
    _ff("-f", "lavfi", "-i", "color=c=green:size=640x360:rate=15", "-t", "0.5", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(halves / "green.mp4"))                      # shorter than the cut-away: last frame holds
    clip = {"start": 0.0, "end": 3.0, "title": "t"}
    render.render_clip(halves / "halves.mp4", halves / "g.mp4", clip, TR, BASE, halves / "w6",
                       edits={"broll": [{"path": str(halves / "green.mp4"), "at": 0.5, "dur": 1.5}]})
    f = _frame(halves / "g.mp4", 1.8, halves)
    assert f[640, 360, 1] > 100 and f[640, 360, 0] < 60 and f[640, 360, 2] < 60   # lavfi "green" is (0,128,0)


# ------------------------------------------------------------------ B-roll matching

from app.pipeline import broll


def test_broll_matches_filenames_to_spoken_words(tmp_path):
    for name in ("city_traffic_night.mp4", "Beaches/sunset-beach.jpg", "notes.txt", "kitchenCooking.mov", "misc/xyz.mp4"):
        f = tmp_path / name
        f.parent.mkdir(exist_ok=True)
        f.write_bytes(b"x")
    items = broll.scan(tmp_path)
    assert {i["name"] for i in items} == {"city_traffic_night.mp4", "sunset-beach.jpg", "kitchenCooking.mov", "xyz.mp4"}
    words = [{"start": 5.0, "end": 5.3, "word": "Traffic"}, {"start": 6.0, "end": 6.3, "word": "beaches"},
             {"start": 7.0, "end": 7.3, "word": "cooking"}, {"start": 8.0, "end": 8.2, "word": "the"}]
    hits = broll.suggest(items, words)
    assert [h["name"] for h in hits] == ["city_traffic_night.mp4", "sunset-beach.jpg", "kitchenCooking.mov"]
    assert hits[0]["matched"] == ["traffic"] and hits[0]["at"] == 5.0
    assert "tokens" not in hits[0]


def test_broll_paths_cannot_escape_the_folder(tmp_path):
    root = tmp_path / "footage"
    root.mkdir()
    (root / "a.mp4").write_bytes(b"x")
    (tmp_path / "secret.mp4").write_bytes(b"x")
    assert broll.is_inside(root / "a.mp4", root)
    assert not broll.is_inside(root / ".." / "secret.mp4", root)
    assert not broll.is_inside(tmp_path / "secret.mp4", root)


@needs_ffmpeg
def test_broll_thumbnails_for_video_and_image(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    monkeypatch.setattr(broll, "BROLL_CACHE", tmp_path / "cache")
    cv2.imwrite(str(tmp_path / "i.png"), np.full((90, 160, 3), 128, np.uint8))
    _ff("-f", "lavfi", "-i", "testsrc=size=160x90:rate=10", "-t", "0.5", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(tmp_path / "short.mp4"))                                           # shorter than the 1 s seek
    assert broll.thumb(tmp_path / "i.png").exists()
    assert broll.thumb(tmp_path / "short.mp4").exists()
    assert broll.thumb(tmp_path / "nope.mp4") is None


# ------------------------------------------------------------------ the API

import json
import zipfile
import io

from app import db, jobs
from app.config import PROJECTS_DIR


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("api")
    mp.setattr(db, "DB_PATH", root / "api.db")
    from app import main
    mp.setattr(main, "KITS_DIR", root / "kits")
    mp.setattr(main, "BRAND_DIR", root / "brand")
    (root / "kits").mkdir()
    (root / "brand").mkdir()
    db.init_db()
    yield TestClient(main.app)
    mp.undo()


@pytest.fixture
def renders(monkeypatch):
    calls = []
    monkeypatch.setattr(jobs, "enqueue_render", lambda pid, ids, ov: calls.append((pid, ids, ov)))
    return calls


@pytest.fixture
def project(halves):
    """A finished project with one clip and a real mp4, cleaned up afterwards."""
    pid = "p3test"
    pdir = PROJECTS_DIR / pid
    (pdir / "clips").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(halves / "halves.mp4", pdir / "clips" / "clip_01.mp4")
    (pdir / "transcript.json").write_text(json.dumps(TR))
    db.execute("DELETE FROM projects WHERE id = ?", (pid,))
    db.execute("INSERT INTO projects(id, title, status, video_file, duration) VALUES(?,?,?,?,?)",
               (pid, "My: show/1", "done", "source.mp4", 60.0))
    cid = db.execute('INSERT INTO clips(project_id, start, "end", title, score, file) VALUES(?,?,?,?,?,?)',
                     (pid, 0.0, 30.0, "Great *clip*", 90, "clip_01.mp4"))
    yield pid, cid
    db.execute("DELETE FROM clips WHERE project_id = ?", (pid,))
    db.execute("DELETE FROM projects WHERE id = ?", (pid,))
    shutil.rmtree(pdir, ignore_errors=True)


def test_new_settings_are_validated(client, tmp_path):
    for bad in ({"export_quality": "ultra"}, {"export_width": "4000"}, {"broll_folder": str(tmp_path / "nope")}):
        assert client.put("/api/settings", json=bad).status_code == 400, bad
    assert client.put("/api/settings", json={"export_quality": "small", "export_width": "720",
                                             "broll_folder": str(tmp_path)}).status_code == 200
    s = client.get("/api/settings").json()["settings"]
    assert s["export_quality"] == "small" and s["broll_folder"] == str(tmp_path)
    client.put("/api/settings", json={"export_quality": "high", "export_width": "1080", "broll_folder": ""})


def test_brand_kits_save_apply_and_feed_renders(client, project, renders):
    pid, cid = project
    client.put("/api/settings", json={"caption_style": "karaoke", "caption_color": "#112233"})
    kid = client.post("/api/kits", json={"name": "  Acme  "}).json()["id"]
    client.put("/api/settings", json={"caption_style": "bold", "caption_color": "#FFFFFF"})
    kits = client.get("/api/kits").json()
    assert [k["name"] for k in kits] == ["Acme"] and kits[0]["opts"]["caption_style"] == "karaoke"
    assert client.post("/api/kits", json={"name": "  "}).status_code == 400
    # a kit applied to one render: its style is used, explicit choices still win
    assert client.post(f"/api/projects/{pid}/render", json={"kit_id": kid, "opts": {"caption_color": "#00FF00"}}).status_code == 200
    ov = renders[-1][2]
    assert ov["caption_style"] == "karaoke" and ov["caption_color"] == "#00FF00"
    assert client.post(f"/api/projects/{pid}/render", json={"kit_id": 999}).status_code == 404
    # applying a kit makes it the default for new projects
    assert client.post(f"/api/kits/{kid}/apply").status_code == 200
    s = client.get("/api/settings").json()["settings"]
    assert s["caption_style"] == "karaoke" and s["caption_color"] == "#112233"
    assert client.delete(f"/api/kits/{kid}").status_code == 200
    assert client.get("/api/kits").json() == []
    client.put("/api/settings", json={"caption_style": "bold", "caption_color": "#FFFFFF"})


def test_kit_carries_its_logo(client, project, renders, tmp_path):
    import cv2
    import numpy as np
    cv2.imwrite(str(tmp_path / "logo.png"), np.full((20, 40, 4), 200, np.uint8))
    assert client.post("/api/brand/logo", files={"file": ("logo.png", (tmp_path / "logo.png").read_bytes(), "image/png")}).status_code == 200
    kid = client.post("/api/kits", json={"name": "WithLogo"}).json()["id"]
    assert client.get("/api/kits").json()[0]["has_logo"] is True
    client.delete("/api/brand/logo")                                   # kit keeps its own copy
    client.post(f"/api/projects/{project[0]}/render", json={"kit_id": kid})
    wm = renders[-1][2]["watermark_path"]
    assert wm.endswith(".png") and Path(wm).exists()
    assert client.get(f"/api/kits/{kid}/logo").status_code == 200
    client.delete(f"/api/kits/{kid}")
    assert not Path(wm).exists()


from pathlib import Path


def test_clip_editor_endpoints(client, project, renders):
    pid, cid = project
    got = client.get(f"/api/clips/{cid}").json()
    assert got["clip"]["title"] == "Great *clip*" and got["project"]["video_file"] == "source.mp4"
    assert got["style"]["caption_style"] and got["clip"]["edits"] == {}
    w = client.get(f"/api/clips/{cid}/words?pad=5").json()["words"]
    assert [x["k"] for x in w][:2] == ["0.20", "0.70"]
    assert "words" not in client.get(f"/api/projects/{pid}").json()["clips"][0]

    ok = client.put(f"/api/clips/{cid}", json={"title": "New", "start": 1.0, "end": 20.0,
                                               "edits": {"cuts": ["0.70"], "words": {"0.20": "Hi"},
                                                         "crop": {"cx": 0.3, "cy": 0.5, "zoom": 1.4}},
                                               "opts": {"caption_style": "pop"}})
    assert ok.status_code == 200 and renders[-1][:2] == (pid, [cid])
    row = client.get(f"/api/clips/{cid}").json()
    assert row["clip"]["start"] == 1.0 and row["clip"]["edits"]["cuts"] == ["0.70"]
    assert row["style"]["caption_style"] == "pop"
    assert client.put(f"/api/clips/{cid}", json={"edits": {}}).status_code == 200
    assert client.get(f"/api/clips/{cid}").json()["clip"]["edits"] == {}


def test_clip_editor_rejects_bad_input(client, project, renders):
    pid, cid = project
    n = len(renders)
    bad = [{"start": 10, "end": 11}, {"start": 0, "end": 500}, {"start": -2, "end": 20}, {"start": 0, "end": 70},
           {"edits": {"crop": {"cx": 2, "cy": 0.5, "zoom": 1}}}, {"edits": {"crop": {"cx": 0.5, "cy": 0.5, "zoom": 9}}},
           {"edits": {"cuts": "abc"}}, {"edits": {"words": {"0.2": 5}}},
           {"edits": {"broll": [{"path": "/etc/passwd", "at": 1, "dur": 2}]}}, {"opts": {"caption_style": "wild"}}]
    for body in bad:
        assert client.put(f"/api/clips/{cid}", json=body).status_code == 400, body
    assert client.put("/api/clips/99999", json={}).status_code == 404
    db.execute("UPDATE projects SET status='running' WHERE id=?", (pid,))
    assert client.put(f"/api/clips/{cid}", json={"title": "x"}).status_code == 409
    db.execute("UPDATE projects SET status='done' WHERE id=?", (pid,))
    assert len(renders) == n


def test_broll_endpoints_stay_inside_the_folder(client, project, renders, tmp_path):
    pid, cid = project
    assert client.get("/api/broll/list").status_code == 400              # no folder yet
    foot = tmp_path / "foot"
    foot.mkdir()
    (foot / "big_world.mp4").write_bytes(b"x")
    (tmp_path / "outside.mp4").write_bytes(b"x")
    client.put("/api/settings", json={"broll_folder": str(foot)})
    assert [i["name"] for i in client.get("/api/broll/list").json()] == ["big_world.mp4"]
    sug = client.get(f"/api/broll/suggest?clip_id={cid}").json()["suggestions"]
    assert sug[0]["name"] == "big_world.mp4" and sug[0]["at"] == pytest.approx(0.7, abs=0.01)
    assert client.get("/api/broll/thumb", params={"path": str(tmp_path / "outside.mp4")}).status_code == 403
    assert client.get("/api/broll/thumb", params={"path": str(foot / ".." / "outside.mp4")}).status_code == 403
    item = {"path": str(foot / "big_world.mp4"), "at": 2, "dur": 3}
    assert client.put(f"/api/clips/{cid}", json={"edits": {"broll": [item]}}).status_code == 200
    assert client.put(f"/api/clips/{cid}", json={"edits": {"broll": [{**item, "dur": 40}]}}).status_code == 400
    client.put("/api/settings", json={"broll_folder": ""})


def test_srt_zip_and_export_folder(client, project, tmp_path):
    pid, cid = project
    srt = client.get(f"/api/clips/{cid}/srt")                          # falls back to the transcript
    assert srt.status_code == 200 and "hello" in srt.text and "-->" in srt.text
    assert 'filename="clip_01.srt"' in srt.headers["content-disposition"]
    db.execute("UPDATE clips SET words=? WHERE id=?", (json.dumps([{"start": 0.0, "end": 0.5, "word": "Saved"}]), cid))
    assert "Saved" in client.get(f"/api/clips/{cid}/srt").text         # uses the final captions once rendered

    z = client.get(f"/api/projects/{pid}/zip")
    assert z.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert names == ["01_Great clip.mp4", "01_Great clip.srt"]

    out = tmp_path / "exports" / "week1"
    r = client.post(f"/api/projects/{pid}/export-folder", json={"folder": str(out)}).json()
    assert r["count"] == 1 and (out / "01_Great clip.mp4").exists() and (out / "01_Great clip.srt").exists()
    assert client.post(f"/api/projects/{pid}/export-folder", json={"folder": "  "}).status_code == 400
    assert client.get("/api/projects/nope/zip").status_code == 404
    db.execute("UPDATE clips SET error='boom' WHERE id=?", (cid,))
    assert client.get(f"/api/projects/{pid}/zip").status_code == 404   # failed clips are not exported


def test_queue_order_retry_and_cancel_all(client, monkeypatch):
    calls = []
    monkeypatch.setattr(jobs, "enqueue", lambda pid, prov=None, from_stage="ingest", focus=None: calls.append((pid, from_stage)))
    cancelled = []
    monkeypatch.setattr(jobs, "cancel", cancelled.append)
    for pid, st, q in (("qa", "queued", 300), ("qb", "running", 100), ("qc", "queued", 200), ("qd", "failed", 0), ("qe", "done", 0)):
        db.execute("INSERT OR REPLACE INTO projects(id, title, status, queued_at) VALUES(?,?,?,?)", (pid, pid, st, q))
    pos = {p["id"]: p["queue_position"] for p in client.get("/api/projects").json() if p["id"].startswith("q")}
    assert pos == {"qa": 2, "qb": 0, "qc": 1, "qd": None, "qe": None}
    assert [q["id"] for q in client.get("/api/queue").json()] == ["qb", "qc", "qa"]
    assert client.post("/api/queue/retry-failed").json() == {"retried": 1} and calls == [("qd", "ingest")]
    assert client.post("/api/queue/cancel-all").json() == {"cancelled": 3} and set(cancelled) == {"qa", "qb", "qc"}
    for pid in ("qa", "qb", "qc", "qd", "qe"):
        db.execute("DELETE FROM projects WHERE id=?", (pid,))


def test_a_project_cancelled_while_waiting_never_starts(client):
    db.execute("INSERT OR REPLACE INTO projects(id, title, status, video_file) VALUES('cw','cw','queued','nothing.mp4')")
    jobs.cancel("cw")
    jobs._run("cw", None, "ingest")
    row = db.one("SELECT status, error FROM projects WHERE id='cw'")
    assert row["status"] == "cancelled"
    db.execute("DELETE FROM projects WHERE id='cw'")
    shutil.rmtree(PROJECTS_DIR / "cw", ignore_errors=True)


def test_health_report_lists_the_essentials(client):
    h = client.get("/api/health").json()
    ids = {c["id"] for c in h["checks"]}
    assert {"ffmpeg", "ffprobe", "nvenc", "cuda", "faster_whisper", "cv2", "disk", "ollama", "broll"} <= ids
    assert all({"label", "ok", "detail", "hint"} <= c.keys() for c in h["checks"])


def test_rerun_and_create_pass_the_focus_through(client, project, monkeypatch):
    seen = []
    monkeypatch.setattr(jobs, "enqueue", lambda pid, prov=None, from_stage="ingest", focus=None: seen.append((from_stage, focus)))
    pid, _ = project
    client.post(f"/api/projects/{pid}/rerun", json={"focus": "pricing"})
    client.post(f"/api/projects/{pid}/rerun", json={})
    assert seen == [("highlights", "pricing"), ("highlights", None)]
    r = client.post("/api/projects", data={"url": "https://example.com/v", "focus": "jokes"})
    assert r.status_code == 200 and seen[-1] == ("ingest", "jokes") or seen[-1][1] == "jokes"
    db.execute("DELETE FROM projects WHERE id=?", (r.json()["id"],))
    shutil.rmtree(PROJECTS_DIR / r.json()["id"], ignore_errors=True)
