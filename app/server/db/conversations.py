"""CRUD cho conversations và messages — SQLite hoặc PostgreSQL."""

from __future__ import annotations

import json
import uuid
from typing import Any

from db.connection import _now, _use_postgres, get_pool, get_sqlite_conn


async def save_turn(
    conv_id: str | None,
    question: str,
    answer: str,
    sources: list[dict],
    chat_type: str = "ask",
    persona_slug: str = "",
    user_id: str | None = None,
) -> str:
    now = _now()
    if _use_postgres():
        return await _save_turn_pg(conv_id, question, answer, sources, chat_type, persona_slug, user_id, now)
    return _save_turn_sqlite(conv_id, question, answer, sources, chat_type, persona_slug, user_id, now)


def _save_turn_sqlite(conv_id, question, answer, sources, chat_type, persona_slug, user_id, now) -> str:
    with get_sqlite_conn() as conn:
        if conv_id:
            row = conn.execute(
                "SELECT id FROM conversations WHERE id=? AND (user_id=? OR user_id IS NULL)",
                (conv_id, user_id),
            ).fetchone()
        else:
            row = None
        if row is None:
            conv_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO conversations(id,user_id,title,chat_type,persona_slug,message_count,preview,created_at,updated_at) "
                "VALUES(?,?,?,?,?,0,'',?,?)",
                (conv_id, user_id, question[:80], chat_type, persona_slug, now, now),
            )
        conn.execute(
            "INSERT INTO messages(conversation_id,role,content,sources_json,created_at) VALUES(?,?,?,?,?)",
            (conv_id, "user", question, "[]", now),
        )
        conn.execute(
            "INSERT INTO messages(conversation_id,role,content,sources_json,created_at) VALUES(?,?,?,?,?)",
            (conv_id, "assistant", answer, json.dumps(sources, ensure_ascii=False), now),
        )
        count = conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id=?", (conv_id,)).fetchone()[0]
        conn.execute(
            "UPDATE conversations SET message_count=?, preview=?, updated_at=? WHERE id=?",
            (count, answer[:200], now, conv_id),
        )
    return conv_id


async def _save_turn_pg(conv_id, question, answer, sources, chat_type, persona_slug, user_id, now) -> str:
    async with get_pool().acquire() as conn:
        async with conn.transaction():
            if conv_id:
                row = await conn.fetchrow(
                    "SELECT id FROM conversations WHERE id=$1 AND (user_id=$2 OR user_id IS NULL)",
                    conv_id, user_id,
                )
            else:
                row = None
            if row is None:
                conv_id = str(uuid.uuid4())
                await conn.execute(
                    "INSERT INTO conversations(id,user_id,title,chat_type,persona_slug,message_count,preview,created_at,updated_at) "
                    "VALUES($1,$2,$3,$4,$5,0,'',$6,$7)",
                    conv_id, user_id, question[:80], chat_type, persona_slug, now, now,
                )
            await conn.execute(
                "INSERT INTO messages(conversation_id,role,content,sources_json,created_at) VALUES($1,$2,$3,$4,$5)",
                conv_id, "user", question, "[]", now,
            )
            await conn.execute(
                "INSERT INTO messages(conversation_id,role,content,sources_json,created_at) VALUES($1,$2,$3,$4,$5)",
                conv_id, "assistant", answer, json.dumps(sources, ensure_ascii=False), now,
            )
            count = await conn.fetchval("SELECT COUNT(*) FROM messages WHERE conversation_id=$1", conv_id)
            await conn.execute(
                "UPDATE conversations SET message_count=$1, preview=$2, updated_at=$3 WHERE id=$4",
                count, answer[:200], now, conv_id,
            )
    return conv_id


async def list_conversations(user_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    if _use_postgres():
        async with get_pool().acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM conversations WHERE user_id=$1 ORDER BY updated_at DESC LIMIT $2",
                user_id, limit,
            )
        return [dict(r) for r in rows]
    with get_sqlite_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM conversations WHERE user_id=? ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


async def get_messages(conv_id: str, user_id: str | None = None) -> dict[str, Any] | None:
    if _use_postgres():
        async with get_pool().acquire() as conn:
            conv = await conn.fetchrow(
                "SELECT * FROM conversations WHERE id=$1 AND (user_id=$2 OR user_id IS NULL)",
                conv_id, user_id,
            )
            if conv is None:
                return None
            msgs = await conn.fetch(
                "SELECT * FROM messages WHERE conversation_id=$1 ORDER BY created_at", conv_id,
            )
        return {**dict(conv), "messages": [{**dict(m), "sources": json.loads(m["sources_json"])} for m in msgs]}

    with get_sqlite_conn() as conn:
        conv = conn.execute(
            "SELECT * FROM conversations WHERE id=? AND (user_id=? OR user_id IS NULL)",
            (conv_id, user_id),
        ).fetchone()
        if conv is None:
            return None
        msgs = conn.execute(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at", (conv_id,)
        ).fetchall()
    return {**dict(conv), "messages": [{**dict(m), "sources": json.loads(m["sources_json"])} for m in msgs]}


async def delete_conversation(conv_id: str, user_id: str | None = None) -> bool:
    if _use_postgres():
        async with get_pool().acquire() as conn:
            result = await conn.execute(
                "DELETE FROM conversations WHERE id=$1 AND (user_id=$2 OR user_id IS NULL)",
                conv_id, user_id,
            )
        return result == "DELETE 1"
    with get_sqlite_conn() as conn:
        cur = conn.execute(
            "DELETE FROM conversations WHERE id=? AND (user_id=? OR user_id IS NULL)",
            (conv_id, user_id),
        )
    return cur.rowcount > 0


async def get_recent_turns(conv_id: str, max_turns: int = 10) -> str:
    if not conv_id:
        return ""
    if _use_postgres():
        async with get_pool().acquire() as conn:
            rows = await conn.fetch(
                "SELECT role, content FROM messages WHERE conversation_id=$1 ORDER BY created_at DESC LIMIT $2",
                conv_id, max_turns * 2,
            )
    else:
        with get_sqlite_conn() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY created_at DESC LIMIT ?",
                (conv_id, max_turns * 2),
            ).fetchall()
    if not rows:
        return ""
    lines = []
    for r in reversed(rows):
        prefix = "Người dùng" if r["role"] == "user" else "Trợ lý"
        lines.append(f"{prefix}: {r['content'][:300]}")
    return "\n".join(lines)


async def get_recent_turns_list(conv_id: str, max_turns: int = 5) -> list[dict]:
    """Trả về danh sách dict {role, content} của max_turns lượt gần nhất (đã đảo về đúng thứ tự)."""
    if not conv_id:
        return []
    if _use_postgres():
        async with get_pool().acquire() as conn:
            rows = await conn.fetch(
                "SELECT role, content FROM messages WHERE conversation_id=$1 ORDER BY created_at DESC LIMIT $2",
                conv_id, max_turns * 2,
            )
    else:
        with get_sqlite_conn() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY created_at DESC LIMIT ?",
                (conv_id, max_turns * 2),
            ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
