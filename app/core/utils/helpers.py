"""Tiện ích dùng chung: xử lý token, định dạng ngữ cảnh và tóm tắt hội thoại."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable, Awaitable

_logger = logging.getLogger(__name__)


# ── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Chuẩn hoá Unicode, xoá ký tự lỗi, gộp khoảng trắng thừa."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\n", "\r")
    text = re.sub(r"[^\S\n]+", " ", text)
    cleaned_lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        line = re.sub(r"[^\w\s.,;:!?%()\[\]/\"'\-–—À-ỹ]", "", line)
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        cleaned_lines.append(line)
    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()


def clean_documents(documents: list) -> list:
    """Áp dụng clean_text lên toàn bộ danh sách Document."""
    cleaned_docs = []
    for doc in documents:
        cleaned_text = clean_text(doc.page_content)
        if cleaned_text:
            doc.page_content = cleaned_text
            cleaned_docs.append(doc)
    print(f"Sau khi clean: {len(cleaned_docs)} documents")
    return cleaned_docs


# ── Lexical Scoring ───────────────────────────────────────────────────────────

_STOPWORDS: frozenset[str] = frozenset({
    "ai", "là", "người", "ra", "của", "và", "với", "trong", "cho",
    "từ", "đến", "có", "này", "đó", "một", "các", "những", "theo",
    "được", "đã", "sẽ", "không", "khi", "vào", "về", "đây", "như",
    "mà", "hay", "thì", "để", "bởi", "vì", "nên", "nhưng", "hoặc",
    "tại", "bị", "do", "rất", "hơn", "cũng", "còn", "đều", "lại",
    "lên", "xuống", "đi", "trên", "dưới", "sau", "trước", "cả",
    "chỉ", "vẫn", "ngay", "thế", "nào", "gì", "đâu", "sao", "ta",
    "ông", "bà", "ấy", "họ", "chúng", "tôi", "em", "anh", "chị",
})


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _extract_tokens(text: str) -> list[str]:
    text = _normalize(text)
    tokens = re.findall(r"[^\W\d_]{2,}", text, re.UNICODE)
    return [t for t in tokens if t not in _STOPWORDS]


def build_query(query: str) -> dict:
    """Xây dựng dict {dense, sparse, keywords} từ câu truy vấn."""
    clean = _normalize(query)
    tokens = _extract_tokens(clean)
    return {
        "dense": clean,
        "sparse": " ".join(tokens[:20]),
        "keywords": tokens,
    }


def _lexical_score(keywords: list[str], content: str) -> float:
    """Tính điểm từ vựng normalize theo độ dài document.

    match / sqrt(len) loại bỏ lợi thế tự nhiên của doc dài;
    coverage * 3.0 thưởng khi câu hỏi được bao phủ tốt.
    """
    content_tokens = set(re.findall(r"[^\W\d_]{2,}", _normalize(content), re.UNICODE))
    match = sum(1 for k in keywords if k in content_tokens)
    length_penalty = max(1.0, len(content_tokens) ** 0.5)
    coverage = match / len(keywords) if keywords else 0
    return (match / length_penalty) * 1.2 + coverage * 3.0


# ── Context Formatting ────────────────────────────────────────────────────────

def format_context_items(items: list[dict[str, Any]]) -> str:
    """Định dạng đoạn văn bản thành chuỗi ngữ cảnh có đánh số nguồn [nguon=i]."""
    sections = []
    for i, item in enumerate(items, 1):
        source_label = item.get("source_label")
        header = f"[nguon={i}]"
        if source_label:
            header = f"{header} {source_label}"
        sections.append(f"{header}\n{item['text']}")
    return "\n\n---\n\n".join(sections)


def split_blocks(text: str) -> list[str]:
    """Tách văn bản thành các đoạn, phân cách bởi dòng trắng."""
    return [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]


def coerce_text(value: Any) -> str:
    """Ép kiểu về str: nếu đã là str trả về nguyên, ngược lại serialize JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def build_source_label(item: dict[str, Any]) -> str:
    """Tạo nhãn nguồn dạng 'tên_file, trang X'."""
    raw_source = (item.get("source") or "").strip()
    file_name = Path(raw_source).name if raw_source else "Tài liệu không rõ tên"
    page_label = item.get("page_label")
    if page_label:
        return f"{file_name}, trang {page_label}"
    page = item.get("page")
    if isinstance(page, int):
        return f"{file_name}, trang {page + 1}"
    return file_name


def build_source_payload(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Xây dựng danh sách metadata nguồn để trả về cùng câu trả lời."""
    sources = []
    for index, item in enumerate(items, start=1):
        raw_source = (item.get("source") or "").strip()
        file_name = Path(raw_source).name if raw_source else ""
        sources.append({
            "index": index,
            "title": item.get("title") or file_name or "Tài liệu không rõ tên",
            "file_name": file_name,
            "file_path": raw_source,
            "page": item.get("page"),
            "page_label": item.get("page_label"),
            "parent_id": item.get("parent_id"),
            "score": round(float(item.get("score") or 0.0), 4),
            "label": build_source_label(item),
        })
    return sources


# ── History Summarization ─────────────────────────────────────────────────────

from app.core.app_config import get_config as _get_config

_SUMMARIZE_THRESHOLD = _get_config().history.summarize_threshold

_SUMMARY_PROMPT = """\
Dưới đây là đoạn hội thoại giữa người dùng và trợ lý về lịch sử Việt Nam.
Hãy viết một đoạn tóm tắt NGẮN GỌN (tối đa 200 từ) bằng tiếng Việt, giữ lại đầy đủ:
- Tên nhân vật lịch sử được nhắc đến (vua, tướng, triều đại, v.v.)
- Sự kiện, năm tháng, địa danh quan trọng
- Mạch chủ đề của cuộc trò chuyện

Quan trọng: nếu có đại từ nhân xưng trong các câu trả lời trước, hãy thay thế bằng tên cụ thể \
của nhân vật đó để tránh mơ hồ.

Chỉ viết đoạn tóm tắt, không thêm tiêu đề hay giải thích.

ĐOẠN HỘI THOẠI:
{dialogue}

TÓM TẮT:"""


def _build_dialogue_text(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        prefix = "Người dùng" if t["role"] == "user" else "Trợ lý"
        lines.append(f"{prefix}: {t['content']}")
    return "\n".join(lines)


async def summarize_turns(
    turns: list[dict],
    llm_func: Callable[[str], Awaitable[str]],
) -> str:
    """Gọi LLM để tóm tắt danh sách turns; trả về chuỗi rỗng nếu lỗi."""
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
    """Tạo khối lịch sử để inject vào prompt chính.

    Nếu ít hơn threshold → inject thẳng.
    Nếu đủ threshold → tóm tắt phần cũ, giữ nguyên recency_turns lượt gần nhất.
    """
    if not turns:
        return ""

    n_pairs = len(turns) // 2
    if n_pairs < _SUMMARIZE_THRESHOLD:
        lines = []
        for t in turns:
            prefix = "Người dùng" if t["role"] == "user" else "Trợ lý"
            lines.append(f"{prefix}: {t['content']}")
        return "\n".join(lines)

    recent_raw = turns[-(recency_turns * 2):]
    older = turns[:-(recency_turns * 2)] if len(turns) > recency_turns * 2 else []

    summary = await summarize_turns(older, llm_func) if older else ""
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


__all__ = [
    "_STOPWORDS",
    "_extract_tokens",
    "_lexical_score",
    "_normalize",
    "build_history_block",
    "build_query",
    "build_source_label",
    "build_source_payload",
    "clean_documents",
    "clean_text",
    "coerce_text",
    "format_context_items",
    "split_blocks",
    "summarize_turns",
]
