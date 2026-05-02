"""HTML page routes — thay thế toàn bộ Django views."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from server.persona_data import PERSONAS, ALL_PERSONA_LIST, DEFAULT_PERSONA_SLUG, BOOKS
from server import db as _db
from server.auth import get_current_user

_logger = logging.getLogger(__name__)

router = APIRouter()
templates: Jinja2Templates | None = None


def set_templates(t: Jinja2Templates) -> None:
    global templates
    templates = t


def _render(request: Request, template: str, ctx: dict) -> HTMLResponse:
    assert templates is not None
    ctx.setdefault("leftsidepath", "")
    ctx.setdefault("current_url_name", "")
    ctx["request"] = request
    ctx.setdefault("current_user", get_current_user(request))
    return templates.TemplateResponse(template, ctx)


# ── Pages ─────────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return _render(request, "home.html", {"leftsidepath": "qa", "current_url_name": "home"})


@router.get("/ask", response_class=HTMLResponse)
async def ask_question(request: Request):
    return _render(request, "Ask_question.html", {"leftsidepath": "qa", "current_url_name": "ask_question"})


@router.get("/history", response_class=HTMLResponse)
async def history(request: Request):
    conversations = _db.list_conversations()
    return _render(request, "history.html", {
        "conversations": conversations,
        "leftsidepath": "history",
        "current_url_name": "history",
    })


@router.get("/persona", response_class=RedirectResponse)
async def persona_redirect():
    return RedirectResponse(url=f"/persona/{DEFAULT_PERSONA_SLUG}", status_code=302)


@router.get("/persona/{slug}", response_class=HTMLResponse)
async def persona_chat(request: Request, slug: str):
    persona = PERSONAS.get(slug)
    if persona is None:
        return RedirectResponse(url=f"/persona/{DEFAULT_PERSONA_SLUG}", status_code=302)
    other_personas = [p for p in ALL_PERSONA_LIST if p["slug"] != slug]
    return _render(request, "persona_chat.html", {
        "persona": persona,
        "other_personas": other_personas,
        "leftsidepath": "persona",
        "current_url_name": "persona_chat_slug",
    })


@router.get("/timeline", response_class=HTMLResponse)
async def timeline(request: Request):
    try:
        dynasties = _db.get_dynasties()
    except Exception:
        _logger.warning("Không thể đọc timeline.sqlite3", exc_info=True)
        dynasties = []
    return _render(request, "time_seri.html", {
        "dynasties_json": dynasties,
        "leftsidepath": "timeline",
        "current_url_name": "timeline",
    })


@router.get("/library", response_class=HTMLResponse)
async def library(request: Request):
    return _render(request, "library.html", {
        "books": BOOKS,
        "books_json": json.dumps(BOOKS, ensure_ascii=False),
        "leftsidepath": "library",
        "current_url_name": "library",
    })


@router.get("/register", response_class=HTMLResponse)
async def register(request: Request):
    return _render(request, "register.html", {"leftsidepath": "", "current_url_name": "register"})


@router.get("/login", response_class=RedirectResponse)
async def login():
    return RedirectResponse(url="/register?tab=login", status_code=302)


@router.get("/map", response_class=HTMLResponse)
async def history_map(request: Request):
    return _render(request, "history_map.html", {"leftsidepath": "", "current_url_name": "map"})


@router.get("/detail", response_class=HTMLResponse)
async def article_detail(request: Request):
    return _render(request, "article_detail.html", {"leftsidepath": "", "current_url_name": "detail"})
