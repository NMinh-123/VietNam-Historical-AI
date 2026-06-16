"""Mẫu prompt cho sử gia trung lập và các nhân vật lịch sử persona."""

from __future__ import annotations

import json
import re
from typing import Any

# ── Historian Prompt (chế độ sử gia trung lập) ───────────────────────────────

HISTORIAN_PROMPT = """\
Bạn là một sử gia Việt Nam uyên bác với phong cách kể chuyện lịch sử lôi cuốn, mạch lạc và luôn tôn trọng sự thật khách quan.

▌LỚP KHUNG (Đồ thị tri thức — đã tổng hợp sẵn)
Dùng để: nắm bức tranh tổng thể, nhân quả, chuỗi sự kiện.

[THỰC_THỂ]
{entities}

[QUAN_HỆ]
{relations}

▌LỚP BẰNG CHỨNG (Văn bản gốc từ sách sử)

[VĂN_BẢN_GỐC]
{vector_context}

━━━ QUY TẮC TRẢ LỜI ━━━

1. PHONG CÁCH HÀNH VĂN:
   - Viết thành đoạn văn liền mạch, chuyển ý tự nhiên (hạn chế gạch đầu dòng trừ khi liệt kê sự kiện độc lập).
   - Dùng LỚP KHUNG để dựng mạch truyện tổng thể; dùng LỚP BẰNG CHỨNG để bổ sung chi tiết.
   - Tuyệt đối không thêm [nguon=#] hay bất kỳ nhãn trích dẫn nào vào câu trả lời.

2. BẢO VỆ SỰ THẬT:
   - Không suy diễn ngoài tài liệu. Nếu thiếu thông tin, ghi nhẹ nhàng: "Tuy nhiên, tài liệu hiện tại chưa ghi rõ...".

{history_block}Câu hỏi của người dùng: {question}
Câu trả lời của Sử gia:
"""

# ── Persona Prompt (nhân vật lịch sử) ────────────────────────────────────────

PERSONA_PROMPT = """\
{system_prompt}

▌TRI THỨC NỀN (đồ thị tri thức — dùng để định hướng câu chuyện)

[THỰC_THỂ]
{entities}

[QUAN_HỆ]
{relations}

▌SỬ LIỆU CỤ THỂ (văn bản gốc)
{vector_context}

━━━ QUY TẮC KHI TRẢ LỜI ━━━

1. NHẬP VAI HOÀN TOÀN:
   - Trả lời với tư cách là {display_name}, không phá vỡ nhân vật.
   - Dùng đúng xưng hô và ngữ điệu đã quy định trong phần mô tả nhân vật trên.
   - Gắn kết câu trả lời với trải nghiệm, ký ức cá nhân của nhân vật khi có thể.

2. GIỚI HẠN THỜI GIAN (BẮT BUỘC):
   - Kiến thức giới hạn đến năm {knowledge_cutoff_year}.
   - Nếu câu hỏi nhắc đến sự kiện/công nghệ/người sau năm {knowledge_cutoff_year}, hãy thừa nhận điều đó vượt khỏi thời đại của mình.

3. PHONG CÁCH HÀNH VĂN:
   - Viết thành đoạn văn liền mạch, có hồn, mang dấu ấn cá nhân của nhân vật.
   - Hạn chế gạch đầu dòng — thay bằng ngôn ngữ kể chuyện tự nhiên.
   - Không bịa đặt sự kiện ngoài sử liệu — nếu chưa đủ thông tin, hãy nói thật với người hỏi.
   - Tuyệt đối không thêm [nguon=#] hay bất kỳ nhãn trích dẫn nào vào câu trả lời.

{history_block}Câu hỏi: {question}
Câu trả lời của {display_name}:
"""

# ── Query Rewriting ───────────────────────────────────────────────────────────

_META_INSTRUCTIONS = (
    "tóm tắt", "tóm lược", "giải thích", "phân tích", "so sánh",
    "liệt kê", "hãy cho biết", "hãy nêu", "hãy trình bày",
    "cho tôi biết", "cho mình biết", "mô tả", "trình bày",
    "kể về", "nói về", "cho biết về", "hỏi về",
    "hãy kể", "hãy mô tả", "hãy phân tích", "hãy so sánh",
    "hãy giải thích", "hãy liệt kê", "làm rõ",
)

_CAUSAL_PATTERNS = [
    (re.compile(r"^lý do\s+(?:dẫn đến|khiến|làm cho|gây ra|của|cho)\s+", re.I), ""),
    (re.compile(r"^nguyên nhân\s+(?:dẫn đến|của|gây ra|khiến|làm cho)?\s*", re.I), ""),
    (re.compile(r"^tại sao\s+", re.I), ""),
    (re.compile(r"^vì sao\s+", re.I), ""),
    (re.compile(r"^do đâu\s+", re.I), ""),
    (re.compile(r"^ảnh hưởng của\s+", re.I), ""),
    (re.compile(r"^hậu quả (?:của|từ)\s+", re.I), ""),
    (re.compile(r"^vai trò (?:của|trong)\s+", re.I), ""),
    (re.compile(r"^ý nghĩa (?:của|lịch sử của)?\s*", re.I), ""),
    (re.compile(r"^quá trình\s+", re.I), ""),
    (re.compile(r"^diễn biến (?:của\s+)?", re.I), ""),
    (re.compile(r"^kết quả (?:của\s+)?", re.I), ""),
]

_LEADING_PREPS = re.compile(r"^(?:của|về|cho|với|trong|từ|đến|là)\s+", re.I)

_TRAILING_NOISE = re.compile(
    r"\s+(?:(?:do|bởi|và|được)\s+)?(?:được\s+\w+\s+)?(?:"
    r"là gì|như thế nào|ra sao|ở đâu|bao giờ|khi nào"
    r"|vào năm nào|năm bao nhiêu|vào ngày nào|vào dịp nào"
    r"|(?:và\s+)?ai\s+(?:đọc|viết|lãnh đạo|chỉ huy|sáng lập|thành lập|là người\s+\w+)"
    r"|do ai\s+\w+"
    r"|có (?:gì|những gì|điểm gì|ý nghĩa gì|vai trò gì|thành tựu gì|đặc điểm gì)"
    r"|diễn ra (?:như thế nào|ra sao|trong hoàn cảnh nào|vào các năm nào)"
    r"|(?:có những?\s+)?(?:cải cách|thành tựu|đặc trưng|điểm tiến bộ) gì"
    r").*$",
    re.I | re.UNICODE,
)


def rewrite_query(question: str) -> str:
    """Tái viết câu hỏi: xoá noise meta-instruction, giữ thực thể lịch sử."""
    q = question.strip()
    q_lower = q.lower()
    for phrase in _META_INSTRUCTIONS:
        if q_lower.startswith(phrase):
            q = q[len(phrase):].lstrip(" ,:")
            break
    for pattern, replacement in _CAUSAL_PATTERNS:
        new_q = pattern.sub(replacement, q)
        if new_q != q:
            q = new_q.strip()
            break
    q = _LEADING_PREPS.sub("", q).strip()
    q = q.strip("?").strip()
    stripped = _TRAILING_NOISE.sub("", q).strip()
    if stripped and len(stripped) >= 5:
        q = stripped
    return q if q else question


# ── Broad Query Detection ─────────────────────────────────────────────────────

_BROAD_PATTERNS = re.compile(
    r"(tất cả|toàn bộ|các triều đại|lịch sử việt nam|"
    r"từ.*đến|xuyên suốt|toàn lịch sử|tổng quan|tổng hợp|"
    r"các thời kỳ|các giai đoạn|nhìn lại|bức tranh|toàn cảnh)",
    re.I | re.UNICODE,
)

BROAD_TOP_K = 12
BROAD_GRAPH_TOP_K = 20

DYNASTIES = [
    "Hồng Bàng Hùng Vương",
    "Triệu Đà Nam Việt",
    "Ngô Quyền nhà Ngô",
    "nhà Đinh Đinh Bộ Lĩnh",
    "nhà Tiền Lê Lê Đại Hành",
    "nhà Lý Lý Thái Tổ Thăng Long",
    "nhà Trần kháng chiến Nguyên Mông",
    "nhà Hồ Hồ Quý Ly",
    "nhà Lê sơ Lê Lợi Lam Sơn",
    "nhà Mạc Mạc Đăng Dung",
    "Trịnh Nguyễn phân tranh Đàng Trong Đàng Ngoài",
    "Tây Sơn Quang Trung Nguyễn Huệ",
    "nhà Nguyễn Gia Long triều Nguyễn",
]


def is_broad_query(question: str) -> bool:
    return bool(_BROAD_PATTERNS.search(question))


def decompose_broad_query(base_query: str) -> list[str]:
    """Ghép từ khoá triều đại với base_query để retrieve có ngữ cảnh."""
    return [f"{dynasty} {base_query}" for dynasty in DYNASTIES]


# ── Knowledge Graph Parsing ───────────────────────────────────────────────────

def parse_graph(items: list[dict]) -> tuple[str, str]:
    """Phân tách entities và relations từ LightRAG context blocks."""
    entities: list[str] = []
    relations: list[str] = []
    raw = "\n".join(b["text"] for b in items)
    in_e = in_r = False

    for line in raw.split("\n"):
        s = line.strip()
        if "Knowledge Graph Data (Entity)" in s:
            in_e, in_r = True, False
        elif "Knowledge Graph Data (Relationship)" in s:
            in_e, in_r = False, True
        elif "Document Chunks" in s:
            in_e = in_r = False
        elif s.startswith("{") and in_e:
            try:
                obj = json.loads(s.rstrip(","))
                desc = obj.get("description", "").split("<SEP>")[0].strip()[:200]
                entities.append(f"• {obj['entity']}: {desc}")
            except Exception:
                pass
        elif s.startswith("{") and in_r:
            try:
                obj = json.loads(s.rstrip(","))
                relations.append(
                    f"• [{obj['entity1']}] → [{obj['entity2']}]: {obj['description']}"
                )
            except Exception:
                pass

    entities_text = "\n".join(entities[:10]) if entities else "(không có dữ liệu thực thể)"
    relations_text = "\n".join(relations[:20]) if relations else "(không có dữ liệu quan hệ)"
    return entities_text, relations_text


# ── Topic Shift Detection ─────────────────────────────────────────────────────

_STOPWORDS_VI = {
    "là", "của", "và", "trong", "có", "không", "đã", "được", "cho", "với",
    "từ", "đến", "về", "hay", "hoặc", "gì", "nào", "bao", "nhiêu", "thì",
    "mà", "khi", "nếu", "để", "vì", "do", "bởi", "tại", "ra", "vào", "lên",
    "xuống", "đi", "lại", "cũng", "còn", "đây", "đó", "này", "kia", "ấy",
    "một", "hai", "các", "những", "mọi", "ai", "sao", "như", "thế", "nên",
    "hãy", "hơn", "nhất", "rất", "quá", "chỉ", "cùng", "theo", "sau", "trước",
}

_AMBIGUOUS_RE = re.compile(
    r"\b(ông|bà|họ|vị|người|nhân vật|vua|tướng|quân|cuộc|sự kiện|"
    r"triều đại|thời kỳ|giai đoạn|chiến thắng|thất bại|phong trào|"
    r"cuộc khởi nghĩa|cuộc chiến|cuộc kháng chiến)\s*(này|đó|ấy|trên|"
    r"đã nêu|vừa nêu|đề cập|nói trên)\b"
    r"|\b(ông|bà|họ|vị)\s+ấy\b",
    re.I | re.UNICODE,
)

TOPIC_SHIFT_THRESHOLD = 0.12


def _word_set(text: str) -> set[str]:
    return {w for w in re.split(r"\s+", text.lower().strip()) if w and w not in _STOPWORDS_VI}


def detect_topic_shift(question: str, turns: list[dict]) -> bool:
    """Trả True nếu câu hỏi đổi chủ đề so với lịch sử hội thoại."""
    if not turns:
        return False
    if _AMBIGUOUS_RE.search(question):
        return False
    recent_user = " ".join(t["content"] for t in turns[-4:] if t["role"] == "user")
    q_words = _word_set(question)
    h_words = _word_set(recent_user)
    if not q_words or not h_words:
        return False
    overlap = len(q_words & h_words) / len(q_words)
    return overlap < TOPIC_SHIFT_THRESHOLD


def build_retrieval_query(question: str, turns: list[dict]) -> tuple[str, bool]:
    """Pure Python — 0 LLM calls. Trả về (retrieval_query, topic_shifted).

    - Không có history: query = câu hỏi gốc.
    - Đổi chủ đề: query = câu hỏi gốc (lịch sử cũ không liên quan).
    - Cùng chủ đề: query = câu hỏi + recent user msgs (window context).
    """
    if not turns:
        return question, False
    shifted = detect_topic_shift(question, turns)
    if shifted:
        return question, True
    recent_user_qs = [
        t["content"][:120]
        for t in turns[-6:]
        if t["role"] == "user"
    ][-2:]
    parts = [question] + recent_user_qs
    return " ".join(parts), False


__all__ = [
    "BROAD_GRAPH_TOP_K",
    "BROAD_TOP_K",
    "DYNASTIES",
    "HISTORIAN_PROMPT",
    "PERSONA_PROMPT",
    "TOPIC_SHIFT_THRESHOLD",
    "build_retrieval_query",
    "decompose_broad_query",
    "detect_topic_shift",
    "is_broad_query",
    "parse_graph",
    "rewrite_query",
]
