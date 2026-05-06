"""Google OAuth 2.0 — login + callback."""

from __future__ import annotations

import logging
import os
import secrets
from typing import Any

from authlib.integrations.httpx_client import AsyncOAuth2Client
from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import RedirectResponse

import db as _db
from auth.session import _set_session_cookie, _callback_url

_logger = logging.getLogger(__name__)

router = APIRouter()

_GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
_GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")

_GOOGLE_CONF = {
    "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
    "token_endpoint":         "https://oauth2.googleapis.com/token",
    "userinfo_endpoint":      "https://www.googleapis.com/oauth2/v3/userinfo",
    "scopes":                 "openid email profile",
}


@router.get("/auth/google/login")
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


@router.get("/auth/google/callback")
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
            token = await client.fetch_token(_GOOGLE_CONF["token_endpoint"], code=code)
            resp_info = await client.get(_GOOGLE_CONF["userinfo_endpoint"])
            info: dict[str, Any] = resp_info.json()
    except Exception as exc:
        _logger.error("Google OAuth thất bại: %s", exc)
        return RedirectResponse("/register?error=google_failed", status_code=302)

    user = await _db.upsert_oauth_account(
        provider="google",
        provider_user_id=info["sub"],
        email=info.get("email", ""),
        display_name=info.get("name", ""),
        avatar_url=info.get("picture", ""),
    )
    request.session.clear()
    resp = RedirectResponse("/", status_code=302)
    _set_session_cookie(resp, user["id"])
    _logger.info("Google login: %s", user["email"])
    return resp
