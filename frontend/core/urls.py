from django.urls import path  # type: ignore
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('ask/', views.ask_question, name='ask_question'),
    path('history/', views.history, name='history'),
    path('persona/', views.persona_chat, name='persona_chat'),
    path('api/persona-chat/', views.persona_chat_api, name='persona_chat_api'),
    path('timeline/', views.time_seri, name='timeline'),
    path('map/', views.history_map, name='history_map'),
    path('detail/', views.article_detail, name='article_detail'),
]