"""Tiện ích xử lý văn bản: chuẩn hoá, tách token và tính điểm từ vựng."""

from __future__ import annotations

import re

# Stop-word tiếng Việt phổ biến, loại khỏi lexical scoring để giảm nhiễu
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
    # Chuẩn hoá khoảng trắng và chuyển về chữ thường
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _extract_tokens(text: str) -> list[str]:
    # Tách token Unicode giữ nguyên dấu tiếng Việt, lọc stop-word và token quá ngắn
    text = _normalize(text)
    tokens = re.findall(r"[^\W\d_]{2,}", text, re.UNICODE)
    return [t for t in tokens if t not in _STOPWORDS]


def build_query(query: str) -> dict:
    # Xây dựng dict chứa vector dày, vector thưa và danh sách từ khoá từ câu truy vấn
    clean = _normalize(query)
    tokens = _extract_tokens(clean)
    return {
        "dense": clean,
        "sparse": " ".join(tokens[:20]),
        "keywords": tokens,
    }


def _lexical_score(keywords: list[str], content: str) -> float:
    # Tính điểm từ vựng normalize theo độ dài document:
    # - match / sqrt(len) loại bỏ lợi thế tự nhiên của doc dài
    # - coverage * 3.0 thưởng khi câu hỏi được bao phủ tốt
    content_tokens = set(re.findall(r"[^\W\d_]{2,}", _normalize(content), re.UNICODE))
    match = sum(1 for k in keywords if k in content_tokens)
    length_penalty = max(1.0, len(content_tokens) ** 0.5)
    coverage = match / len(keywords) if keywords else 0
    return (match / length_penalty) * 1.2 + coverage * 3.0


__all__ = [
    "_STOPWORDS",
    "_normalize",
    "_extract_tokens",
    "build_query",
    "_lexical_score",
]
