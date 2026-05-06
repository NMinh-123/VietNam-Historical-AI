"""CRUD cho users và oauth_accounts — SQLite hoặc PostgreSQL."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import bcrypt as _bcrypt

from db.connection import _now, _use_postgres, get_pool, get_sqlite_conn


async def get_user_by_email(email: str) -> dict | None:
    if _use_postgres():
        async with get_pool().acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM users WHERE email=$1", email)
        return dict(row) if row else None
    with get_sqlite_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    return dict(row) if row else None


async def get_user_by_id(user_id: str) -> dict | None:
    if _use_postgres():
        async with get_pool().acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM users WHERE id=$1", user_id)
        return dict(row) if row else None
    with get_sqlite_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return dict(row) if row else None


async def create_user(
    email: str,
    display_name: str = "",
    avatar_url: str = "",
    password: str | None = None,
) -> dict:
    now = _now()
    uid = str(uuid.uuid4())
    pw_hash = _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode() if password else None
    if _use_postgres():
        async with get_pool().acquire() as conn:
            await conn.execute(
                "INSERT INTO users(id,email,display_name,avatar_url,password_hash,created_at,updated_at) "
                "VALUES($1,$2,$3,$4,$5,$6,$7)",
                uid, email, display_name, avatar_url, pw_hash, now, now,
            )
    else:
        with get_sqlite_conn() as conn:
            conn.execute(
                "INSERT INTO users(id,email,display_name,avatar_url,password_hash,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (uid, email, display_name, avatar_url, pw_hash, now, now),
            )
    return await get_user_by_id(uid)  # type: ignore[return-value]


async def verify_password(user: dict, password: str) -> bool:
    pw_hash = user.get("password_hash")
    if not pw_hash:
        return False
    return await asyncio.to_thread(_bcrypt.checkpw, password.encode(), pw_hash.encode())


async def upsert_oauth_account(
    provider: str,
    provider_user_id: str,
    email: str,
    display_name: str = "",
    avatar_url: str = "",
) -> dict:
    now = _now()
    if _use_postgres():
        async with get_pool().acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO users(id,email,display_name,avatar_url,created_at,updated_at) "
                    "VALUES($1,$2,$3,$4,$5,$6) ON CONFLICT(email) DO NOTHING",
                    str(uuid.uuid4()), email, display_name, avatar_url, now, now,
                )
                user_id = await conn.fetchval("SELECT id FROM users WHERE email=$1", email)
                await conn.execute(
                    "INSERT INTO oauth_accounts(user_id,provider,provider_user_id,created_at) "
                    "VALUES($1,$2,$3,$4) ON CONFLICT(provider,provider_user_id) DO NOTHING",
                    user_id, provider, provider_user_id, now,
                )
                await conn.execute(
                    "UPDATE users SET "
                    "display_name=CASE WHEN display_name='' THEN $1 ELSE display_name END, "
                    "avatar_url=CASE WHEN avatar_url='' THEN $2 ELSE avatar_url END, "
                    "updated_at=$3 WHERE id=$4",
                    display_name, avatar_url, now, user_id,
                )
    else:
        with get_sqlite_conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO users(id,email,display_name,avatar_url,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?)",
                (str(uuid.uuid4()), email, display_name, avatar_url, now, now),
            )
            user_id = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()["id"]
            conn.execute(
                "INSERT OR IGNORE INTO oauth_accounts(user_id,provider,provider_user_id,created_at) "
                "VALUES(?,?,?,?)",
                (user_id, provider, provider_user_id, now),
            )
            conn.execute(
                "UPDATE users SET display_name=CASE WHEN display_name='' THEN ? ELSE display_name END, "
                "avatar_url=CASE WHEN avatar_url='' THEN ? ELSE avatar_url END, updated_at=? WHERE id=?",
                (display_name, avatar_url, now, user_id),
            )
    return await get_user_by_id(user_id)  # type: ignore[return-value]


async def update_user_profile(user_id: str, display_name: str) -> dict | None:
    now = _now()
    if _use_postgres():
        async with get_pool().acquire() as conn:
            await conn.execute(
                "UPDATE users SET display_name=$1, updated_at=$2 WHERE id=$3",
                display_name.strip(), now, user_id,
            )
    else:
        with get_sqlite_conn() as conn:
            conn.execute(
                "UPDATE users SET display_name=?, updated_at=? WHERE id=?",
                (display_name.strip(), now, user_id),
            )
    return await get_user_by_id(user_id)


async def get_user_stats(user_id: str) -> dict[str, int]:
    if _use_postgres():
        async with get_pool().acquire() as conn:
            conv_count = await conn.fetchval("SELECT COUNT(*) FROM conversations WHERE user_id=$1", user_id)
            msg_count = await conn.fetchval(
                "SELECT COUNT(*) FROM messages WHERE role='user' "
                "AND conversation_id IN (SELECT id FROM conversations WHERE user_id=$1)",
                user_id,
            )
    else:
        with get_sqlite_conn() as conn:
            conv_count = conn.execute("SELECT COUNT(*) FROM conversations WHERE user_id=?", (user_id,)).fetchone()[0]
            msg_count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE role='user' "
                "AND conversation_id IN (SELECT id FROM conversations WHERE user_id=?)",
                (user_id,),
            ).fetchone()[0]
    return {"conversations": conv_count, "messages": msg_count}


async def get_oauth_providers(user_id: str) -> list[str]:
    if _use_postgres():
        async with get_pool().acquire() as conn:
            rows = await conn.fetch("SELECT provider FROM oauth_accounts WHERE user_id=$1", user_id)
        return [r["provider"] for r in rows]
    with get_sqlite_conn() as conn:
        rows = conn.execute("SELECT provider FROM oauth_accounts WHERE user_id=?", (user_id,)).fetchall()
    return [r["provider"] for r in rows]
