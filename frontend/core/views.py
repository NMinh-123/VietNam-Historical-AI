from django.http import JsonResponse  # type: ignore
from django.shortcuts import render  # type: ignore
from django.views.decorators.http import require_POST  # type: ignore
from django.views.decorators.csrf import csrf_exempt  # type: ignore

from pathlib import Path
import json
import sys

# Kết nối tới thư mục backend/data để dùng RAG chatbot
ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DATA_DIR = ROOT_DIR / "backend"
if str(BACKEND_DATA_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DATA_DIR))

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
    API nhận câu hỏi của người dùng và trả lời từ RAG backend.
    Body JSON: { "question": "..." }
    """
    try:
        body = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "Dữ liệu không hợp lệ"}, status=400)

    question = (body.get("question") or "").strip()
    if not question:
        return JsonResponse({"error": "Vui lòng nhập câu hỏi"}, status=400)

    # Import "lazy" để Django không crash khi dependency RAG (langchain/chromadb/httpx)
    # không tương thích môi trường Python hiện tại.
    try:
        from data.rag_ollama import ask_rag  # type: ignore

        result = ask_rag(question)
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        verification = result.get("verification", "")
    except Exception as rag_exc:
        # Fallback: gọi trực tiếp Ollama bằng requests (không cần httpx/chromadb).
        try:
            from data.ollama_client import ask_ollama  # type: ignore

            prompt = f"""
Bạn là "Vical AI" chuyên đối thoại với tiền nhân, trả lời bằng tiếng Việt.
Hãy trả lời như một bậc tiền nhân uyên bác, văn phong trang trọng, mạch lạc.

QUY TẮC:
- Trả lời đúng câu hỏi.
- Không bịa thông tin lịch sử cụ thể nếu không chắc.
- Nếu cần, nêu quan điểm khái quát và khuyến nghị cách vận dụng.

Câu hỏi: {question}

Trả lời:
"""
            answer = ask_ollama(prompt).strip()
            sources = []
            verification = f"RAG không chạy được ({rag_exc!s}), dùng fallback Ollama."
        except Exception as ollama_exc:  # pragma: no cover
            return JsonResponse(
                {"error": f"Lỗi hệ thống chatbot: {ollama_exc!s}"},
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
