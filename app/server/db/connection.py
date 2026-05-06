"""DB connection — SQLite (dev) hoặc PostgreSQL (production via DATABASE_URL)."""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_logger = logging.getLogger(__name__)

_TIMELINE_PATH: Path | None = None

# ── Backend detection ──────────────────────────────────────────────────────────

def _use_postgres() -> bool:
    return bool(os.getenv("DATABASE_URL") or os.getenv("POSTGRES_HOST"))


# ── PostgreSQL (asyncpg) ───────────────────────────────────────────────────────

_pool = None  # asyncpg.Pool


def _pg_dsn() -> str:
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return dsn
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db   = os.getenv("POSTGRES_DB", "vical")
    user = os.getenv("POSTGRES_USER", "vical")
    pw   = os.getenv("POSTGRES_PASSWORD", "vical")
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


def get_pool():
    assert _pool is not None, "PostgreSQL pool chưa khởi tạo"
    return _pool


# ── SQLite ─────────────────────────────────────────────────────────────────────

_DB_PATH: Path | None = None


def _sqlite_conn() -> sqlite3.Connection:
    assert _DB_PATH is not None, "Call init_db() first"
    conn = sqlite3.connect(_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def get_sqlite_conn() -> sqlite3.Connection:
    return _sqlite_conn()


def get_timeline_conn() -> sqlite3.Connection:
    assert _TIMELINE_PATH is not None, "Call init_db() first"
    conn = sqlite3.connect(_TIMELINE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ── Init ───────────────────────────────────────────────────────────────────────

async def init_db(timeline_path: Path, db_path: Path | None = None) -> None:
    global _pool, _DB_PATH, _TIMELINE_PATH
    _TIMELINE_PATH = timeline_path

    if _use_postgres():
        import asyncpg
        _pool = await asyncpg.create_pool(dsn=_pg_dsn(), min_size=2, max_size=10, command_timeout=30)
        await _create_pg_tables()
        _logger.info("PostgreSQL pool khởi tạo thành công.")
    else:
        assert db_path is not None, "db_path required khi dùng SQLite"
        _DB_PATH = db_path
        _create_sqlite_tables()
        _logger.info("SQLite khởi tạo tại %s", db_path)


# ── Schema ─────────────────────────────────────────────────────────────────────

_SCHEMA_SQLITE = """
    CREATE TABLE IF NOT EXISTS users (
        id            TEXT PRIMARY KEY,
        email         TEXT NOT NULL UNIQUE,
        display_name  TEXT NOT NULL DEFAULT '',
        avatar_url    TEXT NOT NULL DEFAULT '',
        password_hash TEXT,
        is_active     INTEGER NOT NULL DEFAULT 1,
        created_at    TEXT NOT NULL,
        updated_at    TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS oauth_accounts (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id          TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        provider         TEXT NOT NULL,
        provider_user_id TEXT NOT NULL,
        created_at       TEXT NOT NULL,
        UNIQUE(provider, provider_user_id)
    );
    CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
    CREATE INDEX IF NOT EXISTS idx_oauth_provider ON oauth_accounts(provider, provider_user_id);
    CREATE TABLE IF NOT EXISTS conversations (
        id            TEXT PRIMARY KEY,
        user_id       TEXT REFERENCES users(id) ON DELETE CASCADE,
        title         TEXT NOT NULL,
        chat_type     TEXT NOT NULL DEFAULT 'ask',
        persona_slug  TEXT NOT NULL DEFAULT '',
        message_count INTEGER NOT NULL DEFAULT 0,
        preview       TEXT NOT NULL DEFAULT '',
        created_at    TEXT NOT NULL,
        updated_at    TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
    CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
    CREATE TABLE IF NOT EXISTS messages (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        role            TEXT NOT NULL,
        content         TEXT NOT NULL,
        sources_json    TEXT NOT NULL DEFAULT '[]',
        created_at      TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
"""

_SCHEMA_PG = """
    CREATE TABLE IF NOT EXISTS users (
        id            TEXT PRIMARY KEY,
        email         TEXT NOT NULL UNIQUE,
        display_name  TEXT NOT NULL DEFAULT '',
        avatar_url    TEXT NOT NULL DEFAULT '',
        password_hash TEXT,
        is_active     INTEGER NOT NULL DEFAULT 1,
        created_at    TEXT NOT NULL,
        updated_at    TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS oauth_accounts (
        id               SERIAL PRIMARY KEY,
        user_id          TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        provider         TEXT NOT NULL,
        provider_user_id TEXT NOT NULL,
        created_at       TEXT NOT NULL,
        UNIQUE(provider, provider_user_id)
    );
    CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
    CREATE INDEX IF NOT EXISTS idx_oauth_provider ON oauth_accounts(provider, provider_user_id);
    CREATE TABLE IF NOT EXISTS conversations (
        id            TEXT PRIMARY KEY,
        user_id       TEXT REFERENCES users(id) ON DELETE CASCADE,
        title         TEXT NOT NULL,
        chat_type     TEXT NOT NULL DEFAULT 'ask',
        persona_slug  TEXT NOT NULL DEFAULT '',
        message_count INTEGER NOT NULL DEFAULT 0,
        preview       TEXT NOT NULL DEFAULT '',
        created_at    TEXT NOT NULL,
        updated_at    TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
    CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
    CREATE TABLE IF NOT EXISTS messages (
        id              SERIAL PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        role            TEXT NOT NULL,
        content         TEXT NOT NULL,
        sources_json    TEXT NOT NULL DEFAULT '[]',
        created_at      TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
"""


def _create_sqlite_tables() -> None:
    with _sqlite_conn() as conn:
        conn.executescript(_SCHEMA_SQLITE)
        _sqlite_migrate(conn)


def _sqlite_migrate(conn: sqlite3.Connection) -> None:
    cols = [r[1] for r in conn.execute("PRAGMA table_info(conversations)").fetchall()]
    if "user_id" not in cols:
        conn.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT REFERENCES users(id) ON DELETE CASCADE")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id)")

    oauth_cols = [r[1] for r in conn.execute("PRAGMA table_info(oauth_accounts)").fetchall()]
    if "access_token" in oauth_cols:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS oauth_accounts_new (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id          TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                provider         TEXT NOT NULL,
                provider_user_id TEXT NOT NULL,
                created_at       TEXT NOT NULL,
                UNIQUE(provider, provider_user_id)
            );
            INSERT INTO oauth_accounts_new(id, user_id, provider, provider_user_id, created_at)
                SELECT id, user_id, provider, provider_user_id, created_at FROM oauth_accounts;
            DROP TABLE oauth_accounts;
            ALTER TABLE oauth_accounts_new RENAME TO oauth_accounts;
            CREATE INDEX IF NOT EXISTS idx_oauth_provider ON oauth_accounts(provider, provider_user_id);
        """)


async def _create_pg_tables() -> None:
    async with get_pool().acquire() as conn:
        await conn.execute(_SCHEMA_PG)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
