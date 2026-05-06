"""Pytest fixtures dùng chung cho toàn bộ test suite."""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile

import pytest

# Thêm app/ vào sys.path để import db, auth, ...
APP_DIR = Path(__file__).resolve().parents[1] / "app"
SERVER_DIR = APP_DIR / "server"
for p in (str(APP_DIR), str(SERVER_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture
def tmp_db(tmp_path: Path):
    """Trả (db_path, timeline_path) — SQLite in-memory thông qua file tạm."""
    import db as _db

    db_path = tmp_path / "test.sqlite3"
    timeline_path = tmp_path / "timeline.sqlite3"

    # Tạo timeline.sqlite3 rỗng với schema tối thiểu
    import sqlite3
    conn = sqlite3.connect(timeline_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS core_dynasty (
            id INTEGER PRIMARY KEY, name TEXT, "order" INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS core_king (
            id INTEGER PRIMARY KEY, dynasty_id INTEGER, name TEXT, "order" INTEGER DEFAULT 0
        );
    """)
    conn.commit()
    conn.close()

    _db.init_db(db_path, timeline_path)
    return db_path, timeline_path
