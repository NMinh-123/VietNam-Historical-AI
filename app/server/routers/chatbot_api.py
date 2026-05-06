"""Chatbot API routes — /ask, /personas, /api/ask/stream, /health, /warmup."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

import db as _db
from auth import get_current_user
from schemas import AskRequest, AskResponse, PersonaInfo, SourceItem
from services.chatbot.shared_engine import get_engine, get_persona_engine
from services.chatbot.persona_chat import get_persona, ALL_PERSONAS
from services.chatbot.index_and_retrieve.history_summarizer import build_history_block

_logger = logging.getLogger(__name__)

router = APIRouter()

TRIAL_LIMIT = 3
TRIAL_ENABLED = False  # tạm tắt rate limit cho unauthenticated users


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


@router.post("/ask", response_model=AskResponse)
@router.post("/api/ask", response_model=AskResponse)
async def ask(request: Request, body: AskRequest) -> AskResponse:
    """Nhận câu hỏi, trả câu trả lời kèm danh sách nguồn tài liệu."""
    user = await get_current_user(request)
    trial_remaining: int | None = None
    if not user and TRIAL_ENABLED:
        trial_count = request.session.get("trial_count", 0)
        if trial_count >= TRIAL_LIMIT:
            raise HTTPException(
                status_code=403,
                detail={"code": "trial_exceeded", "limit": TRIAL_LIMIT},
            )
        request.session["trial_count"] = trial_count + 1
        trial_remaining = TRIAL_LIMIT - (trial_count + 1)

    turns = await _db.get_recent_turns_list(body.conversation_id) if body.conversation_id else []
    engine = get_engine()
    history = await build_history_block(turns, engine.llm)
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
            result = await get_persona_engine().ask_with_sources(
                body.question, persona=persona, history=history, turns=turns
            )
            slug_used = persona.slug
        else:
            result = await engine.ask_with_sources(body.question, history=history, turns=turns)
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


@router.post("/api/persona-chat/{slug}", response_model=AskResponse)
async def persona_chat_api(slug: str, request: Request, body: AskRequest) -> AskResponse:
    body.persona_slug = slug
    return await ask(request, body)


@router.get("/personas", response_model=list[PersonaInfo])
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


@router.post("/api/ask/stream")
async def ask_stream(request: Request, body: AskRequest) -> StreamingResponse:
    """SSE endpoint: stream token-by-token từ LLM."""
    user = await get_current_user(request)
    if not user and TRIAL_ENABLED:
        trial_count = request.session.get("trial_count", 0)
        if trial_count >= TRIAL_LIMIT:
            raise HTTPException(status_code=403, detail={"code": "trial_exceeded"})
        request.session["trial_count"] = trial_count + 1

    turns = await _db.get_recent_turns_list(body.conversation_id) if body.conversation_id else []
    engine = get_engine()
    history = await build_history_block(turns, engine.llm)

    async def event_generator():
        import json as _json
        try:
            if body.persona_slug:
                persona = get_persona(body.persona_slug)
                if persona is None:
                    yield f"data: {_json.dumps({'type': 'error', 'message': 'Persona không tồn tại.'})}\n\n"
                    return
                result = await get_persona_engine().ask_with_sources(
                    body.question, persona=persona, history=history, turns=turns
                )
                yield f"data: {_json.dumps({'type': 'token', 'text': result['answer']})}\n\n"
                yield f"data: {_json.dumps({'type': 'done', 'sources': result.get('sources', [])})}\n\n"
            else:
                async for event in engine.ask_with_sources_stream(
                    body.question, history=history, turns=turns
                ):
                    yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            import json as _json
            yield f"data: {_json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/trial-status")
async def trial_status(request: Request) -> dict:
    user = await get_current_user(request)
    if user:
        return {"authenticated": True, "remaining": None}
    used = request.session.get("trial_count", 0)
    return {"authenticated": False, "remaining": max(0, TRIAL_LIMIT - used)}


@router.get("/health")
async def health() -> dict:
    from services.chatbot.shared_engine import get_engine
    try:
        engine = get_engine()
        qdrant_ok = True
        try:
            engine.qdrant.get_collections()
        except Exception:
            qdrant_ok = False
        status = "ok" if (engine._rag_ready and qdrant_ok) else "degraded"
        return {
            "status": status,
            "rag_ready": engine._rag_ready,
            "qdrant_ok": qdrant_ok,
            "personas": [p.slug for p in ALL_PERSONAS],
        }
    except RuntimeError:
        return {"status": "starting", "rag_ready": False, "qdrant_ok": False}


@router.get("/warmup")
async def warmup() -> dict:
    """Kiểm tra trạng thái engine và force warmup nếu chưa sẵn sàng."""
    try:
        engine = get_engine()
        rag_ready = engine._rag_ready
        if not rag_ready:
            _logger.info("Force warming up engine...")
            await engine._init_rag()
            rag_ready = True

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
            },
        }
    except Exception as exc:
        _logger.error("Warmup failed: %s", exc)
        return {"status": "error", "error": str(exc), "type": type(exc).__name__}
