"""Auth package — re-export router tổng hợp, get_current_user, set_templates."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.templating import Jinja2Templates

from auth.session import get_current_user
from auth.email_auth import router as _email_router
from auth.oauth_google import router as _google_router
from auth.oauth_facebook import router as _fb_router
from auth import account as _account_mod

router = APIRouter(tags=["auth"])
router.include_router(_email_router)
router.include_router(_google_router)
router.include_router(_fb_router)
router.include_router(_account_mod.router)


def set_templates(t: Jinja2Templates) -> None:
    _account_mod.set_templates(t)


__all__ = ["router", "get_current_user", "set_templates"]
