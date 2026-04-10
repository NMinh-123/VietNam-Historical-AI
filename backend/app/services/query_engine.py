import asyncio
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

# Model embedding sparse (BM25-like)
from fastembed import SparseTextEmbedding

# Query config cho LightRAG
from lightrag import QueryParam

# Client kết nối Qdrant (vector DB)
from qdrant_client import QdrantClient, models

# Embedding model E5 (dense embedding)
from data.process_data.e5_embeddings import (
    E5_EMBEDDING_DIM,
    E5_MAX_LENGTH,
    E5_QUERY_PROMPT_NAME,
    E5EmbeddingConfig,
    E5EmbeddingModel,
)

# Import config + helper cho LightRAG + Gemini LLM
from services.lightrag import (
    COLLECTION_NAME,
    LIGHTRAG_WORKSPACE,
    PARENT_DOCSTORE_PATH,
    EmbeddingFunc,
    _build_gemini_llm_func,
    _require_gemini_key,
    _resolve_gemini_model_name,
)

# =========================
# TEXT NORMALIZATION
# =========================

# Chuẩn hóa text: bỏ khoảng trắng thừa + lowercase
def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()

# Bỏ dấu tiếng Việt để phục vụ matching tốt hơn
def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn").replace("đ", "d")

# Token hóa text (chỉ giữ chữ + số >= 2 ký tự)
def _extract_tokens(text: str) -> list[str]:
    text = _strip_accents(_normalize(text))
    return re.findall(r"[a-z0-9]{2,}", text)

# Build query gồm:
# - dense: dùng cho embedding
# - sparse: dùng cho BM25
# - keywords: dùng cho lexical scoring
def build_query(query: str):
    clean = _normalize(query)
    tokens = _extract_tokens(clean)

    return {
        "dense": clean,
        "sparse": " ".join(tokens[:10]),
        "keywords": tokens,
    }

# =========================
# LEXICAL SCORING
# =========================

# Tính điểm lexical (matching keyword đơn giản)
def _lexical_score(keywords: list[str], content: str) -> float:
    content_norm = _strip_accents(_normalize(content))
    tokens = set(re.findall(r"[a-z0-9]+", content_norm))

    # số keyword match
    match = sum(1 for k in keywords if k in tokens)

    # độ phủ keyword
    coverage = match / len(keywords) if keywords else 0

    # score = weighted match + coverage
    return match * 1.2 + coverage * 3.0

# =========================
# FORMAT CONTEXT
# =========================

# Format vector context (có source label)
def _format_context_items(items: list[dict[str, Any]]) -> str:
    sections = []
    for i, item in enumerate(items, 1):
        source_label = item.get("source_label")
        header = f"[nguon={i}]"
        if source_label:
            header = f"{header} {source_label}"
        sections.append(f"{header}\n{item['text']}")
    return "\n\n---\n\n".join(sections)

# Format graph context (không có source)
def _format_graph_context_items(items: list[dict[str, Any]]) -> str:
    sections = []
    for i, item in enumerate(items, 1):
        sections.append(f"[goi_y_do_thi={i}]\n{item['text']}")
    return "\n\n---\n\n".join(sections)

# Tách block text theo paragraph
def _split_blocks(text: str):
    return [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]

# Convert mọi kiểu dữ liệu về string
def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)

# Build label hiển thị nguồn (file + page)
def _build_source_label(item: dict[str, Any]) -> str:
    raw_source = (item.get("source") or "").strip()
    file_name = Path(raw_source).name if raw_source else "Tài liệu không rõ tên"

    page_label = item.get("page_label")
    if page_label:
        return f"{file_name}, trang {page_label}"

    page = item.get("page")
    if isinstance(page, int):
        return f"{file_name}, trang {page + 1}"

    return file_name

# Build metadata trả về cho frontend
def _build_source_payload(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = []
    for index, item in enumerate(items, start=1):
        raw_source = (item.get("source") or "").strip()
        file_name = Path(raw_source).name if raw_source else ""
        sources.append(
            {
                "index": index,
                "title": item.get("title") or file_name or "Tài liệu không rõ tên",
                "file_name": file_name,
                "file_path": raw_source,
                "page": item.get("page"),
                "page_label": item.get("page_label"),
                "parent_id": item.get("parent_id"),
                "score": round(float(item.get("score") or 0.0), 4),
                "label": _build_source_label(item),
            }
        )
    return sources

# =========================
# MAIN QUERY ENGINE
# =========================

class VietnamHistoryQueryEngine:
    def __init__(self):
        print("🚀 Initializing Query Engine...")

        # Load API key + model Gemini
        self.api_key = _require_gemini_key()
        self.llm_model_name = _resolve_gemini_model_name()

        # Kết nối Qdrant
        self.qdrant = QdrantClient(host="localhost", port=6333)

        # Dense embedding model (E5)
        self.dense_model = E5EmbeddingModel(E5EmbeddingConfig(prompt_name=E5_QUERY_PROMPT_NAME))

        # Sparse embedding (BM25)
        self.sparse_model = SparseTextEmbedding("Qdrant/bm25")

        # Load parent document store
        self.parent_store = self._load_parent_store()

        # Khởi tạo LLM (Gemini)
        self.llm = _build_gemini_llm_func(
            gemini_key=self.api_key,
            gemini_model_name=self.llm_model_name,
            requests_per_minute=200,
            max_concurrency=4,
            transient_max_retries=3,
        )

        # Hàm gọi LLM có retry (NOTE: hiện đang bị đặt sai scope)
        async def safe_llm_call(self, prompt: str):
            for _ in range(3):
                try:
                    return await self.llm(prompt)
                except Exception as e:
                    print("Retry LLM...", e)
                return "LLM đang quá tải, vui lòng thử lại."

        from services.lightrag import LightRAG

        # Wrapper embedding function cho LightRAG
        async def embed_func(texts):
            return self.dense_model.embed(texts)

        # Khởi tạo LightRAG
        self.rag = LightRAG(
            working_dir=str(LIGHTRAG_WORKSPACE),
            llm_model_func=self.llm,
            embedding_func=EmbeddingFunc(
                embedding_dim=E5_EMBEDDING_DIM,
                max_token_size=E5_MAX_LENGTH,
                func=embed_func,
            ),
        )

        self._rag_ready = False
        self._lock = asyncio.Lock()

    # Load parent documents
    def _load_parent_store(self):
        try:
            with open(PARENT_DOCSTORE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}

    # Lazy init LightRAG
    async def _init_rag(self):
        if self._rag_ready:
            return
        async with self._lock:
            if not self._rag_ready:
                await self.rag.initialize_storages()
                self._rag_ready = True

    # =========================
    # VECTOR SEARCH + RERANK
    # =========================

    def _retrieve(self, query: str, top_k: int, limit: int):
        q = build_query(query)

        # Dense embedding
        dense_vec = self.dense_model.embed([q["dense"]])[0]

        # Sparse embedding
        sparse_vec = list(self.sparse_model.embed([q["sparse"]]))[0]

        # Hybrid search (dense + sparse + RRF fusion)
        result = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=dense_vec.tolist(), using="dense", limit=limit),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist(),
                    ),
                    using="sparse",
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )

        # Group theo parent_id
        grouped = {}

        for hit in result.points:
            payload = hit.payload or {}
            parent_id = payload.get("parent_id")

            # lấy context từ parent store
            context = self.parent_store.get(parent_id) or payload.get("page_content", "")
            if not context:
                continue

            fused = float(hit.score or 0.0)

            # lexical scoring bổ sung
            lexical = _lexical_score(q["keywords"], context)

            # final score
            score = 0.7 * fused + 0.3 * lexical

            key = parent_id or context[:100]

            # aggregate score theo parent
            if key not in grouped:
                grouped[key] = {
                    "text": context,
                    "score": score,
                    "parent_id": parent_id,
                    "count": 1,
                    "source": payload.get("source"),
                    "page": payload.get("page"),
                    "page_label": payload.get("page_label"),
                    "title": payload.get("title"),
                }
            else:
                grouped[key]["score"] += score
                grouped[key]["count"] += 1

        # sort theo score + frequency
        ranked = sorted(
            grouped.values(),
            key=lambda x: x["score"] + 0.2 * x["count"],
            reverse=True,
        )

        # diversify (giới hạn mỗi parent tối đa 2 lần)
        final = []
        seen = {}

        for item in ranked:
            pid = item["parent_id"]
            if seen.get(pid, 0) < 2:
                item["source_label"] = _build_source_label(item)
                final.append(item)
                seen[pid] = seen.get(pid, 0) + 1
            if len(final) >= top_k:
                break

        return {"items": final}

    # Async wrapper cho retrieval
    async def get_vector(self, query: str, top_k=4, limit=40):
        return await asyncio.to_thread(self._retrieve, query, top_k, limit)

    # =========================
    # GRAPH RETRIEVAL
    # =========================

    async def get_graph(self, query: str, vector_items: list[dict], top_k=10):
        await self._init_rag()

        # seed từ vector context
        seed = " ".join([item["text"][:200] for item in vector_items])
        guided_query = f"{query}\n{seed}"

        try:
            raw = await self.rag.aquery(
                guided_query,
                param=QueryParam(
                    mode="local",
                    only_need_context=True,
                    top_k=top_k,
                    chunk_top_k=0,
                ),
            )
        except:
            return {"items": []}

        text = _coerce_text(raw)

        return {
            "items": [{"text": b} for b in _split_blocks(text)]
        }

    # =========================
    # MAIN PIPELINE
    # =========================

    async def ask_with_sources(self, question: str) -> dict[str, Any]:
        print("\n🚀 Running pipeline...")

        # vector retrieval
        vector_bundle = await self.get_vector(question)
        vector_items = vector_bundle["items"]

        # graph retrieval
        graph_bundle = await self.get_graph(question, vector_items)

        # fallback nếu không có dữ liệu
        if not vector_items and not graph_bundle["items"]:
            return {
                "answer": "Tôi chưa tìm thấy tài liệu lịch sử chính xác về vấn đề này.",
                "sources": [],
                "verification": "Không tìm được nguồn vector phù hợp trong chỉ mục hiện tại.",
            }

        # format context
        vector_context = _format_context_items(vector_items)
        graph_context = _format_graph_context_items(graph_bundle["items"])
        sources = _build_source_payload(vector_items)

        # prompt cho LLM (RAG prompt rất chặt chẽ)
        prompt = f""" 
        Bạn là một sử gia Việt Nam uyên bác. Bạn có phong cách kể chuyện lịch sử lôi cuốn, tự nhiên, mạch lạc nhưng luôn tôn trọng tuyệt đối sự thật khách quan dựa trên các sử liệu được cung cấp.

[TEXT_SOURCES] - Nguồn Dữ Liệu Chính (Chứa các sự kiện, ngày tháng, tên gọi)
{vector_context}

[GRAPH_GUIDANCE] - Ngữ Cảnh Đồ Thị (Giúp hiểu cấu trúc, nguyên nhân, mối quan hệ)
{graph_context}

Nhiệm vụ của bạn là tổng hợp thông tin để trả lời câu hỏi của người dùng một cách trọn vẹn và hấp dẫn nhất. Hãy tuân thủ các quy tắc cốt lõi sau:

1. PHONG CÁCH HÀNH VĂN:
- Viết thành các đoạn văn liền mạch, chuyển ý tự nhiên. Hạn chế lạm dụng gạch đầu dòng trừ khi cần liệt kê các chiến dịch/sự kiện hoàn toàn độc lập.
- Sử dụng từ ngữ nối linh hoạt (ví dụ: "Bước vào giai đoạn này...", "Không dừng lại ở đó...", "Chính vì vậy...").
- Dùng [GRAPH_GUIDANCE] để làm phong phú mạch truyện (hiểu ai có quan hệ với ai, sự kiện nào dẫn đến sự kiện nào), nhưng mọi kết luận khẳng định phải có nền tảng từ [TEXT_SOURCES].

2. QUY TẮC TRÍCH DẪN:
- Khéo léo lồng ghép nhãn [nguon=#] vào cuối câu hoặc ngay sau thông tin quan trọng. Nhãn nguồn phải khớp chính xác với số thứ tự trong [TEXT_SOURCES].
- Ví dụ cách viết tự nhiên: "Để thống nhất đất nước, Đinh Bộ Lĩnh đã áp dụng chiến thuật linh hoạt đối với từng sứ quân [nguon=4]." (Không viết kiểu: "Dựa vào nguồn 4, Đinh Bộ Lĩnh...")

3. BẢO VỆ SỰ THẬT:
- Tuyệt đối không tự suy diễn hoặc lấy kiến thức bên ngoài đưa vào.
- Nếu các nguồn tài liệu không chứa đủ thông tin để trả lời trọn vẹn, hãy trả lời phần có thể và nhẹ nhàng thêm vào: "Tuy nhiên, dựa trên các tài liệu hiện tại, chưa có đủ thông tin chi tiết về..." (Không trả lời cộc lốc "Tôi không biết").

Câu hỏi của người dùng: {question}
Câu trả lời của Sử gia:
"""

        # gọi LLM
        answer = await self.llm(prompt)

        verification = (
            f"Trả lời dựa trên {len(sources)} nguồn vector đã truy hồi; "
            "ngữ cảnh đồ thị chỉ dùng để gợi ý, không dùng làm nguồn độc lập."
        )

        return {
            "answer": answer,
            "sources": sources,
            "verification": verification,
        }

    # API đơn giản (không cần sources)
    async def ask(self, question: str) -> str:
        result = await self.ask_with_sources(question)
        return result["answer"]

# =========================
# TEST CLI
# =========================

async def main():
    engine = VietnamHistoryQueryEngine()

    while True:
        q = input("\n👤 ")
        if q.lower() in ["q", "exit"]:
            break

        ans = await engine.ask(q)
        print("\n🤖", ans)

if __name__ == "__main__":
    asyncio.run(main())