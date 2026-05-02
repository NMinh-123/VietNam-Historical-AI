"""Singleton engine dùng chung cho toàn bộ process.

Cả chatbot và persona_chat đều dùng cùng một instance VietnamHistoryQueryEngine
để tránh load model gấp đôi, tốn RAM.
"""

from __future__ import annotations

import asyncio
import logging
import os

_logger = logging.getLogger(__name__)
_engine = None
_engine_lock = asyncio.Lock()


def get_engine():
    """Trả engine duy nhất; khởi tạo lần đầu khi được gọi lần đầu tiên."""
    global _engine
    if _engine is None:
        from services.chatbot.chatbot import VietnamHistoryQueryEngine

        top_k = int(os.getenv("RETRIEVER_TOP_K", "4"))
        limit = int(os.getenv("RETRIEVER_LIMIT", "40"))
        _logger.info(
            "Khởi tạo VietnamHistoryQueryEngine (top_k=%d, limit=%d)", top_k, limit
        )
        _engine = VietnamHistoryQueryEngine(top_k=top_k, limit=limit)
    return _engine


def get_persona_engine():
    """Trả PersonaChatEngine wrapping engine duy nhất."""
    from services.chatbot.persona_chat import PersonaChatEngine
    return PersonaChatEngine(base_engine=get_engine())
