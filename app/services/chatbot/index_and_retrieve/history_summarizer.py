"""Tóm tắt lịch sử hội thoại bằng LLM để giữ mạch ngữ cảnh qua các lượt."""

from __future__ import annotations

import logging
from typing import Callable, Awaitable

_logger = logging.getLogger(__name__)

# Số lượt tối thiểu trước khi cần tóm tắt (< 4 lượt thì inject thẳng cũng ổn)
_SUMMARIZE_THRESHOLD = 4

_SUMMARY_PROMPT = """\
Dưới đây là đoạn hội thoại giữa người dùng và trợ lý về lịch sử Việt Nam.
Hãy viết một đoạn tóm tắt NGẮN GỌN (tối đa 200 từ) bằng tiếng Việt, \
giữ lại đầy đủ:
- Tên nhân vật lịch sử được nhắc đến (vua, tướng, triều đại, v.v.)
- Sự kiện, năm tháng, địa danh quan trọng
- Mạch chủ đề của cuộc trò chuyện

Quan trọng: nếu có đại từ nhân xưng (ông, bà, ông ấy, bà ấy, họ, v.v.) \
trong các câu trả lời trước, hãy thay thế bằng tên cụ thể của nhân vật đó \
để tránh mơ hồ.

Chỉ viết đoạn tóm tắt, không thêm tiêu đề hay giải thích.

ĐOẠN HỘI THOẠI:
{dialogue}

TÓM TẮT:"""


def _build_dialogue_text(turns: list[dict]) -> str:
    """Chuyển danh sách {role, content} thành văn bản hội thoại để đưa vào prompt."""
    lines = []
    for t in turns:
        prefix = "Người dùng" if t["role"] == "user" else "Trợ lý"
        lines.append(f"{prefix}: {t['content']}")
    return "\n".join(lines)


async def summarize_turns(
    turns: list[dict],
    llm_func: Callable[[str], Awaitable[str]],
) -> str:
    """
    Gọi LLM để tóm tắt danh sách turns.
    Trả về chuỗi tóm tắt, hoặc chuỗi rỗng nếu có lỗi.
    """
    if not turns:
        return ""

    dialogue = _build_dialogue_text(turns)
    prompt = _SUMMARY_PROMPT.format(dialogue=dialogue)
    try:
        summary = await llm_func(prompt)
        return summary.strip()
    except Exception as exc:
        _logger.warning("Không thể tóm tắt lịch sử hội thoại: %s", exc)
        return ""


async def build_history_block(
    turns: list[dict],
    llm_func: Callable[[str], Awaitable[str]],
    recency_turns: int = 2,
) -> str:
    """
    Tạo khối lịch sử để inject vào prompt chính.

    Chiến lược:
    - Nếu turns ít hơn threshold → inject thẳng (không cần tóm tắt).
    - Nếu đủ threshold → tóm tắt toàn bộ turns, thêm 2 lượt gần nhất dạng thô
      để LLM giữ được ngữ cảnh câu hỏi ngay trước đó.

    Trả về chuỗi sẵn sàng để đưa vào history_block của prompt template.
    """
    if not turns:
        return ""

    n_pairs = len(turns) // 2  # mỗi cặp user+assistant = 1 lượt

    if n_pairs < _SUMMARIZE_THRESHOLD:
        # Inject thẳng — đủ ngắn để không tốn nhiều token
        lines = []
        for t in turns:
            prefix = "Người dùng" if t["role"] == "user" else "Trợ lý"
            lines.append(f"{prefix}: {t['content']}")
        return "\n".join(lines)

    # Tóm tắt toàn bộ, giữ lại recency_turns lượt gần nhất dạng thô
    recent_raw = turns[-(recency_turns * 2):]  # 2 lượt = 4 messages
    older = turns[:-(recency_turns * 2)] if len(turns) > recency_turns * 2 else []

    if older:
        summary = await summarize_turns(older, llm_func)
    else:
        summary = ""

    parts: list[str] = []
    if summary:
        parts.append(f"[Tóm tắt ngữ cảnh trước]\n{summary}")
    if recent_raw:
        lines = []
        for t in recent_raw:
            prefix = "Người dùng" if t["role"] == "user" else "Trợ lý"
            lines.append(f"{prefix}: {t['content']}")
        parts.append("[Các lượt gần nhất]\n" + "\n".join(lines))

    return "\n\n".join(parts)
