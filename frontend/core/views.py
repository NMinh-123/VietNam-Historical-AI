from django.http import JsonResponse  # type: ignore
from django.shortcuts import render  # type: ignore
from django.views.decorators.http import require_POST  # type: ignore
from django.views.decorators.csrf import csrf_exempt  # type: ignore
from asgiref.sync import async_to_sync  # type: ignore

from pathlib import Path
import json
import sys

# Kết nối tới package backend/app để dùng chung luồng services.index/query_engine
ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_APP_DIR = ROOT_DIR / "backend" / "app"
if str(BACKEND_APP_DIR) not in sys.path:
    sys.path.append(str(BACKEND_APP_DIR))

_QUERY_ENGINE = None


def _get_query_engine():
    global _QUERY_ENGINE
    if _QUERY_ENGINE is None:
        from services.query_engine import VietnamHistoryQueryEngine  # type: ignore

        _QUERY_ENGINE = VietnamHistoryQueryEngine()
    return _QUERY_ENGINE

def home(request):
    return render(request, 'home.html')

def ask_question(request):
    return render(request, 'Ask_question.html')

def history(request):
    return render(request, 'history.html')

def persona_chat(request):
    return render(request, 'persona_chat.html')


@require_POST
@csrf_exempt  # Nếu sau này bạn bật CSRF đầy đủ có thể chuyển sang dùng token phía client
def persona_chat_api(request):
    """
    API nhận câu hỏi của người dùng và trả lời từ luồng index/Qdrant/LightRAG.
    Body JSON: { "question": "..." }
    """
    try:
        body = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "Dữ liệu không hợp lệ"}, status=400)

    question = (body.get("question") or "").strip()
    if not question:
        return JsonResponse({"error": "Vui lòng nhập câu hỏi"}, status=400)

    try:
        engine = _get_query_engine()
        chat_result = async_to_sync(engine.ask_with_sources)(question)
        answer = chat_result["answer"]
        sources = chat_result.get("sources", [])
        verification = chat_result.get(
            "verification",
            "Luồng trả lời: services.query_engine -> services.index",
        )
    except Exception as exc:  # pragma: no cover
        return JsonResponse(
            {"error": f"Lỗi hệ thống chatbot: {exc!s}"},
            status=500,
        )

    return JsonResponse(
        {
            "answer": answer,
            "sources": sources,
            "verification": verification,
        }
    )

def time_seri(request):
    return render(request, 'time_seri.html')

def history_map(request):
    return render(request, 'history_map.html')

def article_detail(request):
    return render(request, 'article_detail.html')
