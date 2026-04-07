"""
Context helpers cho layout: sidebar trái dùng biến template `leftsidepath`.

`leftsidepath` là slug trang đang active: qa | persona | timeline | history
"""

from __future__ import annotations

# url_name Django -> slug hiển thị active trên sidebar
LEFTSIDE_PATH_BY_URL_NAME: dict[str, str] = {
    "home": "qa",
    "ask_question": "qa",
    "persona_chat": "persona",
    "timeline": "timeline",
    "history": "history",
}


def leftsidepath(request):  # type: ignore[no-untyped-def]
    """Context processor: `leftsidepath` (sidebar) và `current_url_name` (mobile nav, an toàn khi không có route)."""
    name = getattr(getattr(request, "resolver_match", None), "url_name", None)
    slug = LEFTSIDE_PATH_BY_URL_NAME.get(name or "", "")
    return {
        "leftsidepath": slug,
        "current_url_name": name or "",
    }
