"""API chatbot lịch sử Việt Nam — phục vụ trang Ask_question."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from asgiref.sync import async_to_sync  # type: ignore
from django.http import JsonResponse  # type: ignore
from django.views.decorators.csrf import csrf_exempt  # type: ignore
from django.views.decorators.http import require_POST  # type: ignore

# Đảm bảo backend/app nằm trong sys.path
_BACKEND_APP_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_APP_DIR))

_logger = logging.getLogger(__name__)

_engine = None


def _get_engine():
    """Lazy-init engine; khởi tạo một lần duy nhất cho toàn bộ process."""
    global _engine
    if _engine is None:
        from services.chatbot.retrieve_and_query import VietnamHistoryQueryEngine  # type: ignore
        _engine = VietnamHistoryQueryEngine()
    return _engine


@require_POST
@csrf_exempt
def ask(request) -> JsonResponse:
    """POST /api/ask/
    Body JSON : { "question": "..." }
    Response  : { "answer": "...", "sources": [...] }
    """
    # --- Parse body ---
    try:
        body = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse({"error": "Dữ liệu không hợp lệ."}, status=400)

    question = (body.get("question") or "").strip()
    if not question:
        return JsonResponse({"error": "Vui lòng nhập câu hỏi."}, status=400)
    if len(question) > 1000:
        return JsonResponse({"error": "Câu hỏi quá dài (tối đa 1000 ký tự)."}, status=400)

    # --- Gọi pipeline ---
    try:
        engine = _get_engine()
        result = async_to_sync(engine.ask_with_sources)(question)
    except Exception:
        _logger.error("Chatbot pipeline thất bại", exc_info=True)
        return JsonResponse({"error": "Lỗi hệ thống, vui lòng thử lại sau."}, status=500)

    return JsonResponse({
        "answer": result["answer"],
        "sources": result.get("sources", []),
    })
