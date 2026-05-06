"""Account page — xem và cập nhật hồ sơ người dùng."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

import db as _db
from auth.session import get_current_user

router = APIRouter()

_templates: Jinja2Templates | None = None


def set_templates(t: Jinja2Templates) -> None:
    global _templates
    _templates = t


@router.get("/auth/account", response_model=None)
async def account_page(request: Request) -> HTMLResponse | RedirectResponse:
    user = await get_current_user(request)
    if not user:
        return RedirectResponse("/register?tab=login", status_code=302)
    assert _templates is not None
    stats = await _db.get_user_stats(user["id"])
    providers = await _db.get_oauth_providers(user["id"])
    return _templates.TemplateResponse(request, "account.html", {
        "current_user": user,
        "stats": stats,
        "providers": providers,
        "leftsidepath": "account",
        "current_url_name": "account",
    })


@router.post("/auth/account/update")
async def account_update(
    request: Request,
    display_name: str = Form(...),
) -> JSONResponse:
    user = await get_current_user(request)
    if not user:
        return JSONResponse({"error": "Chưa đăng nhập."}, status_code=401)
    if not display_name.strip():
        return JSONResponse({"error": "Tên không được để trống."}, status_code=400)
    updated = await _db.update_user_profile(user["id"], display_name)
    return JSONResponse({"ok": True, "display_name": updated["display_name"]})
