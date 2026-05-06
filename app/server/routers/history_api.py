"""History CRUD API."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Literal

import db as _db
from auth import get_current_user

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/history")


class SaveRequest(BaseModel):
    conversation_id: str | None = None
    question: str
    answer: str
    sources: list[dict] = []
    chat_type: Literal["ask", "persona"] = "ask"
    persona_slug: str | None = None


@router.post("/save")
async def history_save(request: Request, body: SaveRequest) -> dict:
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail={"code": "login_required"})
    if not body.question.strip() or not body.answer.strip():
        raise HTTPException(status_code=400, detail="question và answer là bắt buộc.")
    conv_id = await _db.save_turn(
        conv_id=body.conversation_id,
        question=body.question.strip(),
        answer=body.answer.strip(),
        sources=body.sources,
        chat_type=body.chat_type,
        persona_slug=body.persona_slug or "",
        user_id=user["id"],
    )
    return {"conversation_id": conv_id}


@router.get("/{conv_id}/messages")
async def history_messages(request: Request, conv_id: str) -> dict:
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail={"code": "login_required"})
    result = await _db.get_messages(conv_id, user_id=user["id"])
    if result is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy cuộc hội thoại.")
    return result


@router.post("/{conv_id}/delete")
async def history_delete(request: Request, conv_id: str) -> dict:
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail={"code": "login_required"})
    deleted = await _db.delete_conversation(conv_id, user_id=user["id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Không tìm thấy cuộc hội thoại.")
    return {"ok": True}
