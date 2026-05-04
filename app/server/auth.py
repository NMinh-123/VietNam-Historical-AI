"""OAuth 2.0 + Email/Password auth — Authlib + itsdangerous sessions.

Bảo mật:
- Mật khẩu: bcrypt hash (passlib) — không bao giờ lưu plaintext
- Session: itsdangerous.URLSafeTimedSerializer ký bằng SECRET_KEY — không thể giả mạo
- Session cookie: HttpOnly + SameSite=Lax + Secure (production) — không đọc được từ JS
- OAuth state: UUID ngẫu nhiên lưu trong session — chống CSRF
- Access token OAuth: lưu trong DB server-side, không gửi về client
- Sensitive config: đọc từ env, không hardcode trong source
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import Any

from authlib.integrations.httpx_client import AsyncOAuth2Client
from fastapi import APIRouter, Form, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from server import db as _db

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

_templates: Jinja2Templates | None = None


def set_templates(t: Jinja2Templates) -> None:
    global _templates
    _templates = t

# ── Session ───────────────────────────────────────────────────────────────────
# SECRET_KEY bắt buộc phải đặt trong .env. Nếu thiếu, server log cảnh báo rõ.
_SECRET_KEY = os.getenv("SECRET_KEY", "")
if not _SECRET_KEY:
    _logger.warning(
        "⚠️  SECRET_KEY chưa được đặt trong .env — session sẽ không an toàn. "
        "Hãy tạo key ngẫu nhiên: python -c \"import secrets; print(secrets.token_hex(32))\""
    )
    _SECRET_KEY = secrets.token_hex(32)  # fallback an toàn cho dev, đổi sau restart

_signer = URLSafeTimedSerializer(_SECRET_KEY, salt="vical-session")
SESSION_COOKIE = "vical_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 30  # 30 ngày

# ── OAuth provider config ─────────────────────────────────────────────────────
_GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
_GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
_FACEBOOK_CLIENT_ID     = os.getenv("FACEBOOK_APP_ID", "")
_FACEBOOK_CLIENT_SECRET = os.getenv("FACEBOOK_APP_SECRET", "")

_GOOGLE_CONF = {
    "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
    "token_endpoint":         "https://oauth2.googleapis.com/token",
    "userinfo_endpoint":      "https://www.googleapis.com/oauth2/v3/userinfo",
    "scopes":                 "openid email profile",
}
_FACEBOOK_CONF = {
    "authorization_endpoint": "https://www.facebook.com/v19.0/dialog/oauth",
    "token_endpoint":         "https://graph.facebook.com/v19.0/oauth/access_token",
    "userinfo_endpoint":      "https://graph.facebook.com/me?fields=id,name,email,picture",
    "scopes":                 "email public_profile",
}


# ── Session helpers ───────────────────────────────────────────────────────────

def create_session_token(user_id: str) -> str:
    return _signer.dumps({"uid": user_id})


def decode_session_token(token: str, max_age: int = SESSION_MAX_AGE) -> str | None:
    """Trả user_id nếu token hợp lệ, None nếu hết hạn hoặc bị giả mạo."""
    try:
        data = _signer.loads(token, max_age=max_age)
        return data.get("uid")
    except (SignatureExpired, BadSignature):
        return None


def get_current_user(request: Request) -> dict | None:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    user_id = decode_session_token(token)
    if not user_id:
        return None
    return _db.get_user_by_id(user_id)


def _set_session_cookie(response: RedirectResponse | JSONResponse, user_id: str) -> None:
    token = create_session_token(user_id)
    is_prod = os.getenv("ENV", "dev").lower() == "production"
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=SESSION_MAX_AGE,
        httponly=True,           # JS không đọc được → chống XSS
        samesite="lax",          # chống CSRF
        secure=is_prod,          # HTTPS only khi production
        path="/",
    )


def _clear_session_cookie(response: RedirectResponse) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")


def _callback_url(request: Request, provider: str) -> str:
    # Ưu tiên env var (nếu deploy tự manage)
    base = os.getenv("REDIRECT_BASE_URL", "").rstrip("/")
    if not base:
        # Đọc từ reverse-proxy headers (X-Forwarded-Proto / X-Forwarded-Host)
        proto = request.headers.get("x-forwarded-proto") or request.url.scheme
        host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
        base = f"{proto}://{host}"
    return f"{base}/auth/{provider}/callback"


# ── Email / Password auth ─────────────────────────────────────────────────────

@router.post("/register")
async def register_email(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    display_name: str = Form(default=""),
) -> JSONResponse:
    if len(password) < 8:
        return JSONResponse({"error": "Mật khẩu phải có ít nhất 8 ký tự."}, status_code=400)

    existing = _db.get_user_by_email(email)
    if existing:
        return JSONResponse({"error": "Email đã được đăng ký."}, status_code=409)

    user = _db.create_user(email=email, display_name=display_name, password=password)
    resp = JSONResponse({"ok": True, "redirect": "/"})
    _set_session_cookie(resp, user["id"])
    _logger.info("Đăng ký mới: %s", email)
    return resp


@router.post("/login")
async def login_email(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
) -> JSONResponse:
    user = _db.get_user_by_email(email)
    if not user or not _db.verify_password(user, password):
        return JSONResponse({"error": "Email hoặc mật khẩu không đúng."}, status_code=401)
    if not user["is_active"]:
        return JSONResponse({"error": "Tài khoản đã bị khoá."}, status_code=403)

    resp = JSONResponse({"ok": True, "redirect": "/"})
    _set_session_cookie(resp, user["id"])
    _logger.info("Đăng nhập: %s", email)
    return resp


@router.post("/logout")
@router.get("/logout")
async def logout() -> RedirectResponse:
    resp = RedirectResponse(url="/", status_code=302)
    _clear_session_cookie(resp)
    return resp


# ── Google OAuth ──────────────────────────────────────────────────────────────

@router.get("/google/login")
async def google_login(request: Request) -> RedirectResponse:
    if not _GOOGLE_CLIENT_ID:
        return RedirectResponse("/register?error=google_not_configured", status_code=302)

    state = secrets.token_urlsafe(32)
    request.session["oauth_state"] = state

    client = AsyncOAuth2Client(
        client_id=_GOOGLE_CLIENT_ID,
        redirect_uri=_callback_url(request, "google"),
        scope=_GOOGLE_CONF["scopes"],
    )
    url, _ = client.create_authorization_url(
        _GOOGLE_CONF["authorization_endpoint"],
        state=state,
        access_type="offline",
    )
    return RedirectResponse(url, status_code=302)


@router.get("/google/callback")
async def google_callback(request: Request) -> RedirectResponse:
    if not _GOOGLE_CLIENT_ID:
        return RedirectResponse("/register?error=google_not_configured", status_code=302)

    state_in_session = request.session.pop("oauth_state", None)
    state_in_request = request.query_params.get("state")
    if not state_in_session or state_in_session != state_in_request:
        return RedirectResponse("/register?error=invalid_state", status_code=302)

    code = request.query_params.get("code")
    if not code:
        return RedirectResponse("/register?error=no_code", status_code=302)

    try:
        async with AsyncOAuth2Client(
            client_id=_GOOGLE_CLIENT_ID,
            client_secret=_GOOGLE_CLIENT_SECRET,
            redirect_uri=_callback_url(request, "google"),
        ) as client:
            token = await client.fetch_token(
                _GOOGLE_CONF["token_endpoint"],
                code=code,
            )
            resp_info = await client.get(_GOOGLE_CONF["userinfo_endpoint"])
            info: dict[str, Any] = resp_info.json()
    except Exception as exc:
        _logger.error("Google OAuth thất bại: %s", exc)
        return RedirectResponse("/register?error=google_failed", status_code=302)

    user = _db.upsert_oauth_account(
        provider="google",
        provider_user_id=info["sub"],
        email=info.get("email", ""),
        display_name=info.get("name", ""),
        avatar_url=info.get("picture", ""),
        access_token=token.get("access_token", ""),
    )
    resp = RedirectResponse("/", status_code=302)
    _set_session_cookie(resp, user["id"])
    _logger.info("Google login: %s", user["email"])
    return resp


# ── Facebook OAuth ────────────────────────────────────────────────────────────

@router.get("/facebook/login")
async def facebook_login(request: Request) -> RedirectResponse:
    if not _FACEBOOK_CLIENT_ID:
        return RedirectResponse("/register?error=facebook_not_configured", status_code=302)

    state = secrets.token_urlsafe(32)
    request.session["oauth_state"] = state

    client = AsyncOAuth2Client(
        client_id=_FACEBOOK_CLIENT_ID,
        redirect_uri=_callback_url(request, "facebook"),
        scope=_FACEBOOK_CONF["scopes"],
    )
    url, _ = client.create_authorization_url(
        _FACEBOOK_CONF["authorization_endpoint"],
        state=state,
    )
    return RedirectResponse(url, status_code=302)


@router.get("/facebook/callback")
async def facebook_callback(request: Request) -> RedirectResponse:
    if not _FACEBOOK_CLIENT_ID:
        return RedirectResponse("/register?error=facebook_not_configured", status_code=302)

    state_in_session = request.session.pop("oauth_state", None)
    state_in_request = request.query_params.get("state")
    if not state_in_session or state_in_session != state_in_request:
        return RedirectResponse("/register?error=invalid_state", status_code=302)

    code = request.query_params.get("code")
    if not code:
        return RedirectResponse("/register?error=no_code", status_code=302)

    try:
        async with AsyncOAuth2Client(
            client_id=_FACEBOOK_CLIENT_ID,
            client_secret=_FACEBOOK_CLIENT_SECRET,
            redirect_uri=_callback_url(request, "facebook"),
        ) as client:
            token = await client.fetch_token(
                _FACEBOOK_CONF["token_endpoint"],
                code=code,
            )
            resp_info = await client.get(_FACEBOOK_CONF["userinfo_endpoint"])
            info: dict[str, Any] = resp_info.json()
    except Exception as exc:
        _logger.error("Facebook OAuth thất bại: %s", exc)
        return RedirectResponse("/register?error=facebook_failed", status_code=302)

    avatar = ""
    pic = info.get("picture", {})
    if isinstance(pic, dict):
        avatar = pic.get("data", {}).get("url", "")

    user = _db.upsert_oauth_account(
        provider="facebook",
        provider_user_id=info["id"],
        email=info.get("email", f"fb_{info['id']}@facebook.local"),
        display_name=info.get("name", ""),
        avatar_url=avatar,
        access_token=token.get("access_token", ""),
    )
    resp = RedirectResponse("/", status_code=302)
    _set_session_cookie(resp, user["id"])
    _logger.info("Facebook login: %s", user["email"])
    return resp


# ── Account page ──────────────────────────────────────────────────────────────

@router.get("/account", response_model=None)
async def account_page(request: Request) -> HTMLResponse | RedirectResponse:
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/register?tab=login", status_code=302)
    assert _templates is not None
    stats = _db.get_user_stats(user["id"])
    providers = _db.get_oauth_providers(user["id"])
    return _templates.TemplateResponse("account.html", {
        "request": request,
        "current_user": user,
        "stats": stats,
        "providers": providers,
        "leftsidepath": "account",
        "current_url_name": "account",
    })


@router.post("/account/update")
async def account_update(
    request: Request,
    display_name: str = Form(...),
) -> JSONResponse:
    user = get_current_user(request)
    if not user:
        return JSONResponse({"error": "Chưa đăng nhập."}, status_code=401)
    if not display_name.strip():
        return JSONResponse({"error": "Tên không được để trống."}, status_code=400)
    updated = _db.update_user_profile(user["id"], display_name)
    return JSONResponse({"ok": True, "display_name": updated["display_name"]})
