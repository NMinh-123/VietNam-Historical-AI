"""DB connection — SQLite (dev) hoặc PostgreSQL (production via DATABASE_URL)."""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_logger = logging.getLogger(__name__)

_TIMELINE_PATH: Path | None = None  # chỉ dùng khi SQLite

# ── Backend detection ──────────────────────────────────────────────────────────

def _use_postgres() -> bool:
    return bool(os.getenv("DATABASE_URL") or os.getenv("POSTGRES_HOST"))


# ── PostgreSQL (asyncpg) ───────────────────────────────────────────────────────

_pool = None  # asyncpg.Pool


def _pg_dsn() -> str:
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return dsn
    host = os.getenv("POSTGRES_HOST") or "localhost"
    port = os.getenv("POSTGRES_PORT") or "5432"
    db   = os.getenv("POSTGRES_DB") or "vical"
    user = os.getenv("POSTGRES_USER") or "vical"
    pw   = os.getenv("POSTGRES_PASSWORD") or "vical"
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

async def init_db(timeline_path: Path | None = None, db_path: Path | None = None) -> None:
    global _pool, _DB_PATH, _TIMELINE_PATH

    if _use_postgres():
        import asyncpg
        _pool = await asyncpg.create_pool(dsn=_pg_dsn(), min_size=2, max_size=10, command_timeout=30)
        await _create_pg_tables()
        await _create_pg_timeline_tables()
        _logger.info("PostgreSQL pool khởi tạo thành công (bao gồm timeline).")
    else:
        assert timeline_path is not None, "timeline_path required khi dùng SQLite"
        assert db_path is not None, "db_path required khi dùng SQLite"
        _TIMELINE_PATH = timeline_path
        _init_timeline_tables(timeline_path)
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
    CREATE TABLE IF NOT EXISTS revoked_sessions (
        sid        TEXT PRIMARY KEY,
        revoked_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_revoked_sessions_at ON revoked_sessions(revoked_at);
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
    CREATE TABLE IF NOT EXISTS revoked_sessions (
        sid        TEXT PRIMARY KEY,
        revoked_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_revoked_sessions_at ON revoked_sessions(revoked_at);
"""


_TIMELINE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS core_dynasty (
        id         INTEGER PRIMARY KEY,
        "order"    INTEGER NOT NULL DEFAULT 0,
        name       TEXT    NOT NULL,
        start_year INTEGER,
        end_year   INTEGER,
        description TEXT,
        color      TEXT
    );
    CREATE TABLE IF NOT EXISTS core_king (
        id          INTEGER PRIMARY KEY,
        dynasty_id  INTEGER NOT NULL REFERENCES core_dynasty(id),
        "order"     INTEGER NOT NULL DEFAULT 0,
        name        TEXT    NOT NULL,
        reign_start INTEGER,
        reign_end   INTEGER,
        description TEXT
    );
"""

_TIMELINE_SEED: list[tuple] = [
    # (dynasty_id, order, name, start_year, end_year, description, color)
    (1,  1,  "Thời kỳ Bắc thuộc",         -179, 938,  "Nghìn năm Bắc thuộc — các cuộc khởi nghĩa của Hai Bà Trưng, Lý Bí, Mai Hắc Đế, Phùng Hưng.", "#6B4226"),
    (2,  2,  "Nhà Ngô",                    939,  965,  "Ngô Quyền đánh tan quân Nam Hán trên sông Bạch Đằng (938), khai sinh nền độc lập.", "#8B6914"),
    (3,  3,  "Nhà Đinh",                   968,  980,  "Đinh Tiên Hoàng thống nhất 12 sứ quân, đặt quốc hiệu Đại Cồ Việt.", "#5C4033"),
    (4,  4,  "Nhà Tiền Lê",                980,  1009, "Lê Hoàn đánh bại quân Tống xâm lược, củng cố nền độc lập.", "#7B5E3A"),
    (5,  5,  "Nhà Lý",                     1009, 1225, "Dời đô về Thăng Long (1010), xây dựng nhà nước phong kiến trung ương tập quyền vững mạnh.", "#2E7D32"),
    (6,  6,  "Nhà Trần",                   1225, 1400, "Ba lần đánh bại quân Nguyên Mông (1258, 1285, 1288). Thịnh vượng về văn hóa và quân sự.", "#D4AF37"),
    (7,  7,  "Nhà Hồ",                     1400, 1407, "Hồ Quý Ly cải cách hành chính và kinh tế; sau bị quân Minh xâm lược.", "#795548"),
    (8,  8,  "Thuộc Minh / Khởi nghĩa Lam Sơn", 1407, 1427, "20 năm Bắc thuộc lần thứ tư; Lê Lợi lãnh đạo nghĩa quân Lam Sơn giành lại độc lập.", "#BF360C"),
    (9,  9,  "Nhà Lê sơ",                  1428, 1527, "Thời kỳ hưng thịnh: bộ luật Hồng Đức, mở rộng lãnh thổ về phương Nam.", "#1565C0"),
    (10, 10, "Nhà Mạc & Lê Trung Hưng",    1527, 1592, "Nam Bắc triều phân tranh; nhà Lê được Trịnh Kiểm phục dựng.", "#6A1B9A"),
    (11, 11, "Thời Trịnh – Nguyễn phân tranh", 1600, 1788, "Đàng Trong – Đàng Ngoài chia đôi đất nước gần 200 năm.", "#E65100"),
    (12, 12, "Nhà Tây Sơn",                1788, 1802, "Nguyễn Huệ đại phá 29 vạn quân Thanh, thống nhất đất nước.", "#C62828"),
    (13, 13, "Nhà Nguyễn",                 1802, 1945, "Triều đại phong kiến cuối cùng; thực dân Pháp xâm lược (1858).", "#4E342E"),
    (14, 14, "Việt Nam hiện đại",           1945, 2000, "Cách mạng tháng Tám (1945), hai cuộc kháng chiến, thống nhất (1975), Đổi Mới (1986).", "#B22222"),
]

_KING_SEED: list[tuple] = [
    # (id, dynasty_id, order, name, reign_start, reign_end, description)
    (1,  2,  1, "Ngô Quyền",           939,  944,  "Người khai quốc, chiến thắng Bạch Đằng 938."),
    (2,  3,  1, "Đinh Tiên Hoàng",     968,  979,  "Thống nhất 12 sứ quân, lập quốc Đại Cồ Việt."),
    (3,  4,  1, "Lê Hoàn",             980,  1005, "Chống Tống thắng lợi, ổn định đất nước."),
    (4,  5,  1, "Lý Thái Tổ",          1009, 1028, "Dời đô về Thăng Long, lập triều Lý."),
    (5,  5,  2, "Lý Thái Tông",        1028, 1054, "Mở rộng lãnh thổ, phát triển Phật giáo."),
    (6,  5,  3, "Lý Thánh Tông",       1054, 1072, "Đặt quốc hiệu Đại Việt, chinh phạt Chiêm Thành."),
    (7,  5,  4, "Lý Nhân Tông",        1072, 1127, "Chống Tống thắng lợi (1075–1077), thịnh vượng nhất triều Lý."),
    (8,  6,  1, "Trần Thái Tông",      1225, 1258, "Vị vua đầu triều Trần, lãnh đạo kháng chiến Nguyên lần 1."),
    (9,  6,  2, "Trần Thánh Tông",     1258, 1278, "Cùng Trần Hưng Đạo đánh bại Nguyên Mông lần 2 (1285)."),
    (10, 6,  3, "Trần Nhân Tông",      1278, 1293, "Thắng Nguyên Mông lần 3 (1288); xuất gia lập phái Trúc Lâm."),
    (11, 7,  1, "Hồ Quý Ly",           1400, 1407, "Cải cách táo bạo; mất nước vào tay nhà Minh."),
    (12, 9,  1, "Lê Thái Tổ (Lê Lợi)", 1428, 1433, "Lãnh đạo khởi nghĩa Lam Sơn, khai sáng nhà Lê sơ."),
    (13, 9,  2, "Lê Thánh Tông",       1460, 1497, "Đỉnh cao thịnh trị: bộ luật Hồng Đức, mở rộng lãnh thổ."),
    (14, 12, 1, "Nguyễn Huệ (Quang Trung)", 1788, 1792, "Đại phá quân Thanh Tết Kỷ Dậu (1789), anh hùng dân tộc."),
    (15, 13, 1, "Gia Long (Nguyễn Ánh)", 1802, 1820, "Thống nhất đất nước, lập triều Nguyễn."),
    (16, 13, 2, "Minh Mạng",           1820, 1841, "Trung ương tập quyền mạnh, mở rộng lãnh thổ."),
    (17, 14, 1, "Hồ Chí Minh",         1945, 1969, "Lãnh tụ cách mạng, Chủ tịch đầu tiên nước VNDCCH."),
]


_TIMELINE_SCHEMA_PG = """
    CREATE TABLE IF NOT EXISTS core_dynasty (
        id          INTEGER PRIMARY KEY,
        "order"     INTEGER NOT NULL DEFAULT 0,
        name        TEXT    NOT NULL,
        start_year  INTEGER,
        end_year    INTEGER,
        description TEXT,
        color       TEXT
    );
    CREATE TABLE IF NOT EXISTS core_king (
        id          INTEGER PRIMARY KEY,
        dynasty_id  INTEGER NOT NULL REFERENCES core_dynasty(id),
        "order"     INTEGER NOT NULL DEFAULT 0,
        name        TEXT    NOT NULL,
        reign_start INTEGER,
        reign_end   INTEGER,
        description TEXT
    );
"""


async def _create_pg_timeline_tables() -> None:
    async with get_pool().acquire() as conn:
        await conn.execute(_TIMELINE_SCHEMA_PG)
        count = await conn.fetchval("SELECT COUNT(*) FROM core_dynasty")
        if count == 0:
            await conn.executemany(
                'INSERT INTO core_dynasty (id, "order", name, start_year, end_year, description, color) '
                "VALUES ($1,$2,$3,$4,$5,$6,$7)",
                _TIMELINE_SEED,
            )
            await conn.executemany(
                'INSERT INTO core_king (id, dynasty_id, "order", name, reign_start, reign_end, description) '
                "VALUES ($1,$2,$3,$4,$5,$6,$7)",
                _KING_SEED,
            )
            _logger.info("Timeline seeded vào PostgreSQL.")


def _init_timeline_tables(path: Path) -> None:
    """Tạo schema timeline và seed data nếu DB còn rỗng."""
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_TIMELINE_SCHEMA)
        count = conn.execute("SELECT COUNT(*) FROM core_dynasty").fetchone()[0]
        if count == 0:
            conn.executemany(
                'INSERT INTO core_dynasty (id, "order", name, start_year, end_year, description, color) VALUES (?,?,?,?,?,?,?)',
                _TIMELINE_SEED,
            )
            conn.executemany(
                'INSERT INTO core_king (id, dynasty_id, "order", name, reign_start, reign_end, description) VALUES (?,?,?,?,?,?,?)',
                _KING_SEED,
            )
            conn.commit()
            _logger.info("Timeline DB seeded tại %s", path)
    finally:
        conn.close()


def _create_sqlite_tables() -> None:
    with _sqlite_conn() as conn:
        conn.executescript(_SCHEMA_SQLITE)
        _sqlite_migrate(conn)


def _sqlite_migrate(conn: sqlite3.Connection) -> None:
    cols = [r[1] for r in conn.execute("PRAGMA table_info(conversations)").fetchall()]
    if "user_id" not in cols:
        conn.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT REFERENCES users(id) ON DELETE CASCADE")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id)")

    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "revoked_sessions" not in tables:
        conn.execute(
            "CREATE TABLE revoked_sessions (sid TEXT PRIMARY KEY, revoked_at TEXT NOT NULL)"
        )
        conn.execute("CREATE INDEX idx_revoked_sessions_at ON revoked_sessions(revoked_at)")

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
