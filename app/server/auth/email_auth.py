"""Email/password auth — register, login, logout."""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

import db as _db
from auth.session import _set_session_cookie, _clear_session_cookie

_limiter = Limiter(key_func=get_remote_address)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/auth/register")
@_limiter.limit("5/minute")
async def register_email(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    display_name: str = Form(default=""),
) -> JSONResponse:
    if not _EMAIL_RE.match(email):
        return JSONResponse({"error": "Email không hợp lệ."}, status_code=400)
    if len(password) < 8:
        return JSONResponse({"error": "Mật khẩu phải có ít nhất 8 ký tự."}, status_code=400)

    existing = await _db.get_user_by_email(email)
    if existing:
        return JSONResponse({"error": "Email đã được đăng ký."}, status_code=409)

    user = await _db.create_user(email=email, display_name=display_name, password=password)
    request.session.clear()
    resp = JSONResponse({"ok": True, "redirect": "/"})
    _set_session_cookie(resp, user["id"])
    _logger.info("Đăng ký mới: %s", email)
    return resp


@router.post("/auth/login")
@_limiter.limit("5/minute")
async def login_email(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
) -> JSONResponse:
    user = await _db.get_user_by_email(email)
    if not user or not await _db.verify_password(user, password):
        return JSONResponse({"error": "Email hoặc mật khẩu không đúng."}, status_code=401)
    if not user["is_active"]:
        return JSONResponse({"error": "Tài khoản đã bị khoá."}, status_code=403)

    request.session.clear()
    resp = JSONResponse({"ok": True, "redirect": "/"})
    _set_session_cookie(resp, user["id"])
    _logger.info("Đăng nhập: %s", email)
    return resp


@router.post("/auth/logout")
@router.get("/auth/logout")
async def logout() -> RedirectResponse:
    resp = RedirectResponse(url="/", status_code=302)
    _clear_session_cookie(resp)
    return resp
