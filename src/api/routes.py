"""Tổng hợp tất cả API routes của Vical AI.

Module này re-export các router từ server/ để main.py có một điểm import duy nhất.
app/ phải có trong sys.path trước khi import module này (main.py đảm nhiệm điều đó).
"""

from __future__ import annotations

# ── Page routes (HTML rendering) ─────────────────────────────────────────────
from server.routers import pages as pages_module  # type: ignore[import]

pages_router = pages_module.router
set_pages_templates = pages_module.set_templates

# ── Chatbot API (ask, stream, personas, health) ───────────────────────────────
from server.routers.chatbot_api import router as chatbot_router  # type: ignore[import]

# ── History CRUD API ──────────────────────────────────────────────────────────
from server.routers.history_api import router as history_router  # type: ignore[import]

# ── Auth (email + Google + Facebook OAuth) ────────────────────────────────────
import auth as auth_module  # type: ignore[import]

auth_router = auth_module.router
set_auth_templates = auth_module.set_templates

__all__ = [
    "auth_module",
    "auth_router",
    "chatbot_router",
    "history_router",
    "pages_module",
    "pages_router",
    "set_auth_templates",
    "set_pages_templates",
]
