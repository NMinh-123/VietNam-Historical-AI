"""Facebook OAuth 2.0 — login + callback."""

from __future__ import annotations

import logging
import os
import secrets
from typing import Any

from authlib.integrations.httpx_client import AsyncOAuth2Client
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

import db as _db
from auth.session import _set_session_cookie, _callback_url

_logger = logging.getLogger(__name__)

router = APIRouter()

_FACEBOOK_CLIENT_ID     = os.getenv("FACEBOOK_APP_ID", "")
_FACEBOOK_CLIENT_SECRET = os.getenv("FACEBOOK_APP_SECRET", "")

_FACEBOOK_CONF = {
    "authorization_endpoint": "https://www.facebook.com/v19.0/dialog/oauth",
    "token_endpoint":         "https://graph.facebook.com/v19.0/oauth/access_token",
    "userinfo_endpoint":      "https://graph.facebook.com/me?fields=id,name,email,picture",
    "scopes":                 "email public_profile",
}


@router.get("/auth/facebook/login")
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


@router.get("/auth/facebook/callback")
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
            token = await client.fetch_token(_FACEBOOK_CONF["token_endpoint"], code=code)
            resp_info = await client.get(_FACEBOOK_CONF["userinfo_endpoint"])
            info: dict[str, Any] = resp_info.json()
    except Exception as exc:
        _logger.error("Facebook OAuth thất bại: %s", exc)
        return RedirectResponse("/register?error=facebook_failed", status_code=302)

    avatar = ""
    pic = info.get("picture", {})
    if isinstance(pic, dict):
        avatar = pic.get("data", {}).get("url", "")

    user = await _db.upsert_oauth_account(
        provider="facebook",
        provider_user_id=info["id"],
        email=info.get("email", f"fb_{info['id']}@facebook.local"),
        display_name=info.get("name", ""),
        avatar_url=avatar,
    )
    request.session.clear()
    resp = RedirectResponse("/", status_code=302)
    _set_session_cookie(resp, user["id"])
    _logger.info("Facebook login: %s", user["email"])
    return resp
