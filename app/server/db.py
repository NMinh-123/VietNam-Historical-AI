"""sqlite3 helper — history conversations, timeline dynasties, users & OAuth accounts."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bcrypt as _bcrypt

_HERE = Path(__file__).resolve().parent

_DB_PATH: Path | None = None
_TIMELINE_PATH: Path | None = None


def init_db(db_path: Path, timeline_path: Path) -> None:
    global _DB_PATH, _TIMELINE_PATH
    _DB_PATH = db_path
    _TIMELINE_PATH = timeline_path
    _create_tables()


def _conn() -> sqlite3.Connection:
    assert _DB_PATH is not None, "Call init_db() first"
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _create_tables() -> None:
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id           TEXT PRIMARY KEY,
                email        TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL DEFAULT '',
                avatar_url   TEXT NOT NULL DEFAULT '',
                password_hash TEXT,
                is_active    INTEGER NOT NULL DEFAULT 1,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS oauth_accounts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                provider    TEXT NOT NULL,
                provider_user_id TEXT NOT NULL,
                access_token TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL,
                UNIQUE(provider, provider_user_id)
            );

            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
            CREATE INDEX IF NOT EXISTS idx_oauth_provider ON oauth_accounts(provider, provider_user_id);

            CREATE TABLE IF NOT EXISTS conversations (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL,
                chat_type   TEXT NOT NULL DEFAULT 'ask',
                persona_slug TEXT NOT NULL DEFAULT '',
                message_count INTEGER NOT NULL DEFAULT 0,
                preview     TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                sources_json    TEXT NOT NULL DEFAULT '[]',
                created_at      TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
        """)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_turn(
    conv_id: str | None,
    question: str,
    answer: str,
    sources: list[dict],
    chat_type: str = "ask",
    persona_slug: str = "",
) -> str:
    now = _now()
    with _conn() as conn:
        if conv_id:
            row = conn.execute("SELECT id FROM conversations WHERE id=?", (conv_id,)).fetchone()
        else:
            row = None

        if row is None:
            conv_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO conversations(id,title,chat_type,persona_slug,message_count,preview,created_at,updated_at) "
                "VALUES(?,?,?,?,0,'',?,?)",
                (conv_id, question[:80], chat_type, persona_slug, now, now),
            )

        conn.execute(
            "INSERT INTO messages(conversation_id,role,content,sources_json,created_at) VALUES(?,?,?,?,?)",
            (conv_id, "user", question, "[]", now),
        )
        conn.execute(
            "INSERT INTO messages(conversation_id,role,content,sources_json,created_at) VALUES(?,?,?,?,?)",
            (conv_id, "assistant", answer, json.dumps(sources, ensure_ascii=False), now),
        )
        count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id=?", (conv_id,)
        ).fetchone()[0]
        conn.execute(
            "UPDATE conversations SET message_count=?, preview=?, updated_at=? WHERE id=?",
            (count, answer[:200], now, conv_id),
        )
    return conv_id


def list_conversations(limit: int = 50) -> list[dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_messages(conv_id: str) -> dict[str, Any] | None:
    with _conn() as conn:
        conv = conn.execute("SELECT * FROM conversations WHERE id=?", (conv_id,)).fetchone()
        if conv is None:
            return None
        msgs = conn.execute(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at", (conv_id,)
        ).fetchall()
    return {
        **dict(conv),
        "messages": [
            {**dict(m), "sources": json.loads(m["sources_json"])}
            for m in msgs
        ],
    }


def delete_conversation(conv_id: str) -> bool:
    with _conn() as conn:
        cur = conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
        conn.execute("DELETE FROM messages WHERE conversation_id=?", (conv_id,))
    return cur.rowcount > 0


def get_recent_turns(conv_id: str, max_turns: int = 10) -> str:
    """Trả chuỗi N lượt hội thoại gần nhất để inject vào prompt."""
    if not conv_id:
        return ""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages "
            "WHERE conversation_id=? ORDER BY created_at DESC LIMIT ?",
            (conv_id, max_turns * 2),
        ).fetchall()
    if not rows:
        return ""
    lines = []
    for r in reversed(rows):
        prefix = "Người dùng" if r["role"] == "user" else "Trợ lý"
        lines.append(f"{prefix}: {r['content'][:300]}")
    return "\n".join(lines)


# ── User & OAuth helpers ──────────────────────────────────────────────────────

def get_user_by_email(email: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return dict(row) if row else None


def create_user(
    email: str,
    display_name: str = "",
    avatar_url: str = "",
    password: str | None = None,
) -> dict:
    now = _now()
    uid = str(uuid.uuid4())
    pw_hash = _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode() if password else None
    with _conn() as conn:
        conn.execute(
            "INSERT INTO users(id,email,display_name,avatar_url,password_hash,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (uid, email, display_name, avatar_url, pw_hash, now, now),
        )
    return get_user_by_id(uid)  # type: ignore[return-value]


def verify_password(user: dict, password: str) -> bool:
    pw_hash = user.get("password_hash")
    if not pw_hash:
        return False
    return _bcrypt.checkpw(password.encode(), pw_hash.encode())


def upsert_oauth_account(
    provider: str,
    provider_user_id: str,
    email: str,
    display_name: str = "",
    avatar_url: str = "",
    access_token: str = "",
) -> dict:
    """Tìm hoặc tạo user qua OAuth. Trả về user dict."""
    now = _now()
    with _conn() as conn:
        # Tìm theo oauth account trước
        row = conn.execute(
            "SELECT user_id FROM oauth_accounts WHERE provider=? AND provider_user_id=?",
            (provider, provider_user_id),
        ).fetchone()
        if row:
            user_id = row["user_id"]
            # Cập nhật access_token
            conn.execute(
                "UPDATE oauth_accounts SET access_token=? WHERE provider=? AND provider_user_id=?",
                (access_token, provider, provider_user_id),
            )
        else:
            # Tìm user theo email (có thể đã đăng ký bằng email trước)
            user_row = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
            if user_row:
                user_id = user_row["id"]
            else:
                # Tạo user mới
                user_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO users(id,email,display_name,avatar_url,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?)",
                    (user_id, email, display_name, avatar_url, now, now),
                )
            conn.execute(
                "INSERT INTO oauth_accounts(user_id,provider,provider_user_id,access_token,created_at) "
                "VALUES(?,?,?,?,?)",
                (user_id, provider, provider_user_id, access_token, now),
            )
        # Cập nhật display_name/avatar nếu thiếu
        conn.execute(
            "UPDATE users SET display_name=CASE WHEN display_name='' THEN ? ELSE display_name END, "
            "avatar_url=CASE WHEN avatar_url='' THEN ? ELSE avatar_url END, updated_at=? WHERE id=?",
            (display_name, avatar_url, now, user_id),
        )
    return get_user_by_id(user_id)  # type: ignore[return-value]


def update_user_profile(user_id: str, display_name: str) -> dict | None:
    now = _now()
    with _conn() as conn:
        conn.execute(
            "UPDATE users SET display_name=?, updated_at=? WHERE id=?",
            (display_name.strip(), now, user_id),
        )
    return get_user_by_id(user_id)


def get_user_stats(user_id: str) -> dict[str, int]:
    """Trả thống kê: số cuộc hội thoại, số tin nhắn đã gửi."""
    with _conn() as conn:
        conv_count = conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE id IN "
            "(SELECT DISTINCT conversation_id FROM messages)",
        ).fetchone()[0]
        msg_count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE role='user'",
        ).fetchone()[0]
    return {"conversations": conv_count, "messages": msg_count}


def get_oauth_providers(user_id: str) -> list[str]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT provider FROM oauth_accounts WHERE user_id=?", (user_id,)
        ).fetchall()
    return [r["provider"] for r in rows]


def get_dynasties() -> list[dict[str, Any]]:
    assert _TIMELINE_PATH is not None, "Call init_db() first"
    conn = sqlite3.connect(_TIMELINE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        dynasties = conn.execute("SELECT * FROM core_dynasty ORDER BY \"order\"").fetchall()
        result = []
        for d in dynasties:
            kings = conn.execute(
                "SELECT * FROM core_king WHERE dynasty_id=? ORDER BY \"order\"", (d["id"],)
            ).fetchall()
            result.append({
                **dict(d),
                "kings": [dict(k) for k in kings],
            })
        return result
    finally:
        conn.close()
