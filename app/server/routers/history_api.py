"""History CRUD API — thay thế Django history API views."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from server import db as _db
from server.auth import get_current_user

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/history")


class SaveRequest(BaseModel):
    conversation_id: str | None = None
    question: str
    answer: str
    sources: list[dict] = []
    chat_type: str = "ask"
    persona_slug: str = ""


@router.post("/save")
async def history_save(request: Request, body: SaveRequest) -> dict:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail={"code": "login_required"})
    if not body.question.strip() or not body.answer.strip():
        raise HTTPException(status_code=400, detail="question và answer là bắt buộc.")
    conv_id = _db.save_turn(
        conv_id=body.conversation_id,
        question=body.question.strip(),
        answer=body.answer.strip(),
        sources=body.sources,
        chat_type=body.chat_type,
        persona_slug=body.persona_slug.strip(),
        user_id=user["id"],
    )
    return {"conversation_id": conv_id}


@router.get("/{conv_id}/messages")
async def history_messages(request: Request, conv_id: str) -> dict:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail={"code": "login_required"})
    result = _db.get_messages(conv_id, user_id=user["id"])
    if result is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy cuộc hội thoại.")
    return result


@router.post("/{conv_id}/delete")
async def history_delete(request: Request, conv_id: str) -> dict:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail={"code": "login_required"})
    deleted = _db.delete_conversation(conv_id, user_id=user["id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Không tìm thấy cuộc hội thoại.")
    return {"ok": True}
