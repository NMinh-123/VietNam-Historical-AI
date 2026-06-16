"""Tổng hợp tất cả API routes của Vical AI."""

from __future__ import annotations

from app.api import pages as pages_module

pages_router = pages_module.router
set_pages_templates = pages_module.set_templates

from app.api.chatbot import router as chatbot_router
from app.api.history import router as history_router

from app import auth as auth_module

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
