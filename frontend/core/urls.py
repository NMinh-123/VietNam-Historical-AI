from django.urls import path  # type: ignore
from . import views
import sys
from pathlib import Path

# Thêm backend/app vào sys.path để import api.chatbot
_BACKEND_APP_DIR = Path(__file__).resolve().parents[2] / "backend" / "app"
if str(_BACKEND_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_APP_DIR))

from api import chatbot as chatbot_api  # type: ignore  # noqa: E402

urlpatterns = [
    path('', views.home, name='home'),
    path('ask/', views.ask_question, name='ask_question'),
    path('history/', views.history, name='history'),
    path('persona/', views.persona_chat, name='persona_chat'),
    path('api/persona-chat/', views.persona_chat_api, name='persona_chat_api'),
    path('api/ask/', chatbot_api.ask, name='chatbot_ask'),
    path('timeline/', views.time_seri, name='timeline'),
    path('map/', views.history_map, name='history_map'),
    path('detail/', views.article_detail, name='article_detail'),
]