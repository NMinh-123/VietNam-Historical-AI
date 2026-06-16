"""Session token, cookie helpers, và get_current_user."""

from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

import db as _db

_logger = logging.getLogger(__name__)

_SECRET_KEY = os.getenv("SECRET_KEY", "")
if not _SECRET_KEY:
    _logger.warning(
        "⚠️  SECRET_KEY chưa được đặt trong .env — session sẽ không an toàn. "
        "Hãy tạo key ngẫu nhiên: python -c \"import secrets; print(secrets.token_hex(32))\""
    )
    _SECRET_KEY = secrets.token_hex(32)

_signer = URLSafeTimedSerializer(_SECRET_KEY, salt="vical-session")
SESSION_COOKIE = "vical_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 30  # 30 ngày


def create_session_token(user_id: str) -> str:
    sid = secrets.token_urlsafe(16)
    return _signer.dumps({"uid": user_id, "sid": sid})


def decode_session_token(token: str, max_age: int = SESSION_MAX_AGE) -> tuple[str, str | None] | None:
    """Trả (user_id, sid) nếu token hợp lệ, None nếu hết hạn hoặc bị giả mạo.
    sid là None với token cũ không có session ID."""
    try:
        data = _signer.loads(token, max_age=max_age)
        uid = data.get("uid")
        if not uid:
            return None
        return uid, data.get("sid")
    except (SignatureExpired, BadSignature):
        return None


async def get_current_user(request: Request) -> dict | None:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    result = decode_session_token(token)
    if not result:
        return None
    user_id, sid = result
    if sid and await _db.is_session_revoked(sid):
        return None
    return await _db.get_user_by_id(user_id)


def _set_session_cookie(response: RedirectResponse | JSONResponse, user_id: str) -> None:
    token = create_session_token(user_id)
    is_prod = os.getenv("ENV", "dev").lower() == "production"
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=is_prod,
        path="/",
    )


def _clear_session_cookie(response: RedirectResponse) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")


def _callback_url(request: Request, provider: str) -> str:
    base = os.getenv("REDIRECT_BASE_URL", "").rstrip("/")
    if not base:
        proto = request.headers.get("x-forwarded-proto") or request.url.scheme
        host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
        base = f"{proto}://{host}"
    return f"{base}/auth/{provider}/callback"
