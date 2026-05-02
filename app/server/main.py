"""FastAPI server — phục vụ cả HTML pages (Jinja2) lẫn chatbot API."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

_APP_DIR = Path(__file__).resolve().parents[1]
_SERVER_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _APP_DIR / "static"
_TEMPLATES_DIR = _SERVER_DIR / "templates"
_DB_PATH = _SERVER_DIR / "db.sqlite3"
_TIMELINE_PATH = _APP_DIR.parent / "data" / "timeline.sqlite3"

if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from services.chatbot.shared_engine import get_engine, get_persona_engine
from services.chatbot.persona_chat import (
    get_persona,
    ALL_PERSONAS,
    DEFAULT_PERSONA_SLUG,
)
from server import db
from server.routers import pages as pages_router
from server.routers import history_api as history_router
from server import auth as auth_router

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _logger.info("Khởi tạo database...")
    db.init_db(_DB_PATH, _TIMELINE_PATH)

    _logger.info("Khởi tạo VietnamHistoryQueryEngine...")
    try:
        engine = get_engine()
        await engine.start()
        # Đợi warmup hoàn tất (có timeout 60s để tránh treo server)
        import asyncio as _asyncio
        try:
            await _asyncio.wait_for(engine._warmup_task, timeout=60.0)
            _logger.info("✓ Engine warm-up hoàn tất, sẵn sàng nhận request.")
        except _asyncio.TimeoutError:
            _logger.warning("⏱️  Engine warm-up timeout 60s — sẽ lazy-load khi có request.")
    except Exception as exc:
        _logger.error("⚠️  Không thể khởi tạo engine: %s", exc)
        _logger.warning("Server sẽ chạy nhưng /ask endpoint sẽ lazy-load engine lần đầu (chậm).")

    pages_router.set_templates(templates)
    auth_router.set_templates(templates)

    yield
    _logger.info("Server đang tắt.")


app = FastAPI(
    title="Vical Chatbot API",
    version="3.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

_SESSION_SECRET = os.getenv("SECRET_KEY") or os.urandom(32).hex()
app.add_middleware(SessionMiddleware, secret_key=_SESSION_SECRET, same_site="lax", https_only=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

app.include_router(pages_router.router)
app.include_router(history_router.router)
app.include_router(auth_router.router)


# ── Schema ────────────────────────────────────────────────────────────────────

TRIAL_LIMIT = 3


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    persona_slug: str | None = Field(default=None)
    conversation_id: str | None = Field(default=None)


class SourceItem(BaseModel):
    index: int
    label: str
    score: float
    title: str | None = None
    file_name: str | None = None
    page: int | None = None
    page_label: str | None = None
    parent_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    verification: str | None = None
    persona_slug: str | None = None
    trial_remaining: int | None = None


class PersonaInfo(BaseModel):
    slug: str
    display_name: str
    title: str
    era_label: str
    bio_short: str
    portrait_url: str
    accent_color: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_source_items(raw_sources: list[dict]) -> list[SourceItem]:
    return [
        SourceItem(
            index=s.get("index", i + 1),
            label=s.get("label", ""),
            score=round(float(s.get("score", 0.0)), 4),
            title=s.get("title"),
            file_name=s.get("file_name"),
            page=s.get("page"),
            page_label=s.get("page_label"),
            parent_id=s.get("parent_id"),
        )
        for i, s in enumerate(raw_sources)
    ]


# ── Chatbot API endpoints ─────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse)
@app.post("/api/ask", response_model=AskResponse)
async def ask(request: Request, body: AskRequest) -> AskResponse:
    """Nhận câu hỏi, trả câu trả lời kèm danh sách nguồn tài liệu."""
    user = auth_router.get_current_user(request)
    trial_remaining: int | None = None
    if not user:
        trial_count = request.session.get("trial_count", 0)
        if trial_count >= TRIAL_LIMIT:
            raise HTTPException(
                status_code=403,
                detail={"code": "trial_exceeded", "limit": TRIAL_LIMIT},
            )
        request.session["trial_count"] = trial_count + 1
        trial_remaining = TRIAL_LIMIT - (trial_count + 1)

    history = db.get_recent_turns(body.conversation_id) if body.conversation_id else ""
    try:
        if body.persona_slug:
            persona = get_persona(body.persona_slug)
            if persona is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Không tìm thấy nhân vật '{body.persona_slug}'. "
                        f"Slug hợp lệ: {[p.slug for p in ALL_PERSONAS]}"
                    ),
                )
            persona_engine = get_persona_engine()
            result = await persona_engine.ask_with_sources(body.question, persona=persona, history=history)
            slug_used = persona.slug
        else:
            engine = get_engine()
            result = await engine.ask_with_sources(body.question, history=history)
            slug_used = None
    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        _logger.error("Pipeline thất bại: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {type(exc).__name__}: {exc}")

    return AskResponse(
        answer=result["answer"],
        sources=_build_source_items(result.get("sources", [])),
        verification=result.get("verification"),
        persona_slug=slug_used,
        trial_remaining=trial_remaining,
    )


@app.post("/api/persona-chat/{slug}", response_model=AskResponse)
async def persona_chat_api(slug: str, request: Request, body: AskRequest) -> AskResponse:
    """POST /api/persona-chat/<slug> — compat alias cho chat templates."""
    body.persona_slug = slug
    return await ask(request, body)


@app.get("/personas", response_model=list[PersonaInfo])
async def list_personas() -> list[PersonaInfo]:
    return [
        PersonaInfo(
            slug=p.slug,
            display_name=p.display_name,
            title=p.title,
            era_label=p.era_label,
            bio_short=p.bio_short,
            portrait_url=p.portrait_url,
            accent_color=p.accent_color,
        )
        for p in ALL_PERSONAS
    ]


@app.post("/api/ask/stream")
async def ask_stream(request: Request, body: AskRequest) -> StreamingResponse:
    """SSE endpoint: stream token-by-token từ LLM."""
    user = auth_router.get_current_user(request)
    if not user:
        trial_count = request.session.get("trial_count", 0)
        if trial_count >= TRIAL_LIMIT:
            raise HTTPException(status_code=403, detail={"code": "trial_exceeded"})
        request.session["trial_count"] = trial_count + 1

    history = db.get_recent_turns(body.conversation_id) if body.conversation_id else ""

    async def event_generator():
        import json as _json
        try:
            if body.persona_slug:
                persona = get_persona(body.persona_slug)
                if persona is None:
                    yield f"data: {_json.dumps({'type': 'error', 'message': 'Persona không tồn tại.'})}\n\n"
                    return
                engine = get_persona_engine()
                result = await engine.ask_with_sources(body.question, persona=persona, history=history)
                yield f"data: {_json.dumps({'type': 'token', 'text': result['answer']})}\n\n"
                yield f"data: {_json.dumps({'type': 'done', 'sources': result.get('sources', [])})}\n\n"
            else:
                engine = get_engine()
                async for event in engine.ask_with_sources_stream(body.question, history=history):
                    yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            import json as _json
            yield f"data: {_json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/trial-status")
async def trial_status(request: Request) -> dict:
    user = auth_router.get_current_user(request)
    if user:
        return {"authenticated": True, "remaining": None}
    used = request.session.get("trial_count", 0)
    return {"authenticated": False, "remaining": max(0, TRIAL_LIMIT - used)}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "personas": [p.slug for p in ALL_PERSONAS]}


@app.get("/warmup")
async def warmup() -> dict:
    """Kiểm tra trạng thái engine và force warmup nếu chưa sẵn sàng."""
    try:
        engine = get_engine()

        # Kiểm tra LightRAG đã ready chưa
        rag_ready = engine._rag_ready

        # Nếu chưa ready, force init
        if not rag_ready:
            _logger.info("Force warming up engine...")
            await engine._init_rag()
            rag_ready = True

        # Kiểm tra Qdrant connection
        try:
            collections = engine.qdrant.get_collections()
            qdrant_ok = True
            qdrant_collections = [c.name for c in collections.collections]
        except Exception as qe:
            qdrant_ok = False
            qdrant_collections = []
            _logger.warning("Qdrant check failed: %s", qe)

        return {
            "status": "ready" if (rag_ready and qdrant_ok) else "warming",
            "lightrag_ready": rag_ready,
            "qdrant_ok": qdrant_ok,
            "qdrant_collections": qdrant_collections,
            "models_loaded": {
                "dense": engine.dense_model is not None,
                "sparse": engine.sparse_model is not None,
            }
        }
    except Exception as exc:
        _logger.error("Warmup failed: %s", exc)
        return {
            "status": "error",
            "error": str(exc),
            "type": type(exc).__name__
        }
