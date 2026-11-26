"""Tiny SQLite helper. One connection per call keeps threading simple."""
import sqlite3
from contextlib import contextmanager

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    title TEXT,
    source TEXT,
    status TEXT DEFAULT 'queued',      -- queued | running | done | failed | cancelled
    stage TEXT DEFAULT 'queued',       -- ingest | transcribe | highlights | export
    progress REAL DEFAULT 0,
    error TEXT,
    video_file TEXT,
    duration REAL,
    provider_used TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    start REAL NOT NULL,
    "end" REAL NOT NULL,
    title TEXT,
    score INTEGER,
    reason TEXT,
    file TEXT,
    provider TEXT,
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS kits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    opts TEXT,                         -- JSON of style settings
    logo TEXT                          -- file name of the kit's own logo inside data/kits
);
"""


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# Columns added after Phase 1. Existing databases are upgraded in place, so old projects survive.
MIGRATIONS = {
    "clips": {
        "layout": "TEXT", "encoder": "TEXT", "out_seconds": "REAL", "removed_seconds": "REAL",
        "rendered_at": "INTEGER", "opts": "TEXT", "error": "TEXT",
        "edits": "TEXT", "words": "TEXT",
    },
    "projects": {"focus": "TEXT", "queued_at": "INTEGER"},
}


def init_db() -> None:
    with connect() as c:
        c.executescript(SCHEMA)
        for table, cols in MIGRATIONS.items():
            have = {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}
            for name, kind in cols.items():
                if name not in have:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")


def rows(sql: str, params: tuple = ()) -> list[dict]:
    with connect() as c:
        return [dict(r) for r in c.execute(sql, params).fetchall()]


def one(sql: str, params: tuple = ()) -> dict | None:
    with connect() as c:
        r = c.execute(sql, params).fetchone()
        return dict(r) if r else None


def execute(sql: str, params: tuple = ()) -> int:
    with connect() as c:
        return c.execute(sql, params).lastrowid
