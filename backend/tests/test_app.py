"""App-level tests. Run from the backend folder:  python -m pytest tests -q"""
import sqlite3

import pytest

from app import db, settings_store

PHASE1_SCHEMA = """
CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE projects (id TEXT PRIMARY KEY, title TEXT, source TEXT, status TEXT DEFAULT 'queued',
  stage TEXT DEFAULT 'queued', progress REAL DEFAULT 0, error TEXT, video_file TEXT, duration REAL,
  provider_used TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE clips (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL, start REAL NOT NULL,
  "end" REAL NOT NULL, title TEXT, score INTEGER, reason TEXT, file TEXT, provider TEXT);
"""


def test_phase1_database_upgrades_in_place_and_keeps_its_data(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript(PHASE1_SCHEMA)
    con.execute("INSERT INTO projects(id, title) VALUES('p1', 'old project')")
    con.execute('INSERT INTO clips(project_id, start, "end", title, score, file) VALUES(?,?,?,?,?,?)',
                ("p1", 1.0, 40.0, "old clip", 77, "clip_01.mp4"))
    con.execute("INSERT INTO settings VALUES('llm_provider', 'gemini')")
    con.commit(); con.close()

    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db()
    db.init_db()                                          # running it twice must be harmless

    clip = db.one("SELECT * FROM clips")
    assert clip["title"] == "old clip" and clip["score"] == 77
    assert {"layout", "encoder", "out_seconds", "removed_seconds", "rendered_at", "opts", "error"} <= set(clip)
    assert clip["layout"] is None and clip["error"] is None
    assert settings_store.get_all()["llm_provider"] == "gemini"        # saved choice survives
    assert settings_store.get_all()["caption_style"] == "bold"         # new settings get defaults


def test_settings_validation():
    v = settings_store.validate
    assert v({"caption_style": "bold", "highlight_color": "#FFE600", "silence_gap": "0.8"}) is None
    assert "caption_style" in v({"caption_style": "wobbly"})
    assert "colour" in v({"highlight_color": "yellow"})
    assert "number" in v({"silence_gap": "soon"})
    assert "output_aspect" in v({"output_aspect": "21:9"})


def test_style_keys_all_have_defaults():
    assert set(settings_store.STYLE_KEYS) <= set(settings_store.DEFAULTS)
    assert set(settings_store.CHOICES) <= set(settings_store.DEFAULTS)
