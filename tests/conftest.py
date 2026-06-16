"""Pytest fixtures dùng chung cho toàn bộ test suite."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def tmp_db(tmp_path: Path, monkeypatch):
    """Fixture async: khởi tạo SQLite in-memory thông qua file tạm.
    Force SQLite bằng cách xoá POSTGRES_HOST để tránh dùng DB production."""
    from app import db as _db
    monkeypatch.delenv("POSTGRES_HOST", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    db_path = tmp_path / "test.sqlite3"
    timeline_path = tmp_path / "timeline.sqlite3"

    conn = sqlite3.connect(timeline_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS core_dynasty (
            id INTEGER PRIMARY KEY,
            name TEXT,
            "order" INTEGER DEFAULT 0,
            start_year INTEGER,
            end_year INTEGER,
            description TEXT,
            color TEXT
        );
        CREATE TABLE IF NOT EXISTS core_king (
            id INTEGER PRIMARY KEY,
            dynasty_id INTEGER,
            name TEXT,
            "order" INTEGER DEFAULT 0,
            reign_start INTEGER,
            reign_end INTEGER,
            description TEXT
        );
    """)
    conn.commit()
    conn.close()

    await _db.init_db(timeline_path, db_path)
    return db_path, timeline_path
