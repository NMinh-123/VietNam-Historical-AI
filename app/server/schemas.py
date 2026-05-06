"""Pydantic schemas dùng chung trong toàn bộ server."""

from __future__ import annotations

from pydantic import BaseModel, Field


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
