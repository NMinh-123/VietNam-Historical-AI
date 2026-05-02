"""Engine chatbot sử gia: điều phối vector search + knowledge graph + LLM.

Chỉ xử lý chế độ sử gia trung lập (không persona).
Persona mode được xử lý riêng bởi PersonaChatEngine trong persona_chat/.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from typing import Any

_logger = logging.getLogger(__name__)

from fastembed import SparseTextEmbedding
from lightrag import QueryParam
from qdrant_client import QdrantClient

from data.process_data.e5_embeddings import (
    E5_EMBEDDING_DIM,
    E5_MAX_LENGTH,
    E5_QUERY_PROMPT_NAME,
    E5EmbeddingConfig,
    E5EmbeddingModel,
)
from services.chatbot.index_and_retrieve import (
    GEMINI_OPENAI_BASE_URL,
    LIGHTRAG_WORKSPACE,
    PARENT_DOCSTORE_PATH,
    QDRANT_HOST,
    QDRANT_PORT,
    EmbeddingFunc,
    LightRAG,
    _build_gemini_llm_func,
    _require_gemini_key,
    _resolve_gemini_model_name,
)
from services.chatbot.index_and_retrieve.context_builder import (
    _build_source_payload,
    _coerce_text,
    _format_context_items,
    _split_blocks,
)
from services.chatbot.index_and_retrieve.retriever import get_vector

# ── Query rewriting ──────────────────────────────────────────────────────────
import re as _re

_META_INSTRUCTIONS = (
    "tóm tắt", "tóm lược", "giải thích", "phân tích", "so sánh",
    "liệt kê", "hãy cho biết", "hãy nêu", "hãy trình bày",
    "cho tôi biết", "cho mình biết", "mô tả", "trình bày",
    "kể về", "nói về", "cho biết về", "hỏi về",
    "hãy kể", "hãy mô tả", "hãy phân tích", "hãy so sánh",
    "hãy giải thích", "hãy liệt kê", "làm rõ",
)

_CAUSAL_PATTERNS = [
    (_re.compile(r"^lý do\s+(?:dẫn đến|khiến|làm cho|gây ra|của|cho)\s+", _re.I), ""),
    (_re.compile(r"^nguyên nhân\s+(?:dẫn đến|của|gây ra|khiến|làm cho)?\s*", _re.I), ""),
    (_re.compile(r"^tại sao\s+", _re.I), ""),
    (_re.compile(r"^vì sao\s+", _re.I), ""),
    (_re.compile(r"^do đâu\s+", _re.I), ""),
    (_re.compile(r"^ảnh hưởng của\s+", _re.I), ""),
    (_re.compile(r"^hậu quả (?:của|từ)\s+", _re.I), ""),
    (_re.compile(r"^vai trò (?:của|trong)\s+", _re.I), ""),
    (_re.compile(r"^ý nghĩa (?:của|lịch sử của)?\s*", _re.I), ""),
    (_re.compile(r"^quá trình\s+", _re.I), ""),
    (_re.compile(r"^diễn biến (?:của\s+)?", _re.I), ""),
    (_re.compile(r"^kết quả (?:của\s+)?", _re.I), ""),
]

_LEADING_PREPS = _re.compile(r"^(?:của|về|cho|với|trong|từ|đến|là)\s+", _re.I)


def _rewrite_query(question: str) -> str:
    """Tái viết câu hỏi: xoá noise meta-instruction và giới từ, giữ thực thể lịch sử."""
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
    return q if q else question


# ── Broad query detection ─────────────────────────────────────────────────────
_BROAD_PATTERNS = _re.compile(
    r"(tất cả|toàn bộ|các triều đại|lịch sử việt nam|"
    r"từ.*đến|xuyên suốt|toàn lịch sử|tổng quan|tổng hợp|"
    r"các thời kỳ|các giai đoạn|nhìn lại|bức tranh|toàn cảnh)",
    _re.I | _re.UNICODE,
)
_BROAD_TOP_K = 12
_BROAD_GRAPH_TOP_K = 20


def _is_broad_query(question: str) -> bool:
    """Trả True nếu câu hỏi mang tính tổng hợp/toàn cảnh nhiều triều đại."""
    return bool(_BROAD_PATTERNS.search(question))


# ── Sub-queries triều đại cho decompose ──────────────────────────────────────
_DYNASTIES = [
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


def _decompose_broad_query(base_query: str) -> list[str]:
    """Tạo sub-queries theo từng triều đại để retrieve song song."""
    return _DYNASTIES


# ── parse_graph: trích entity + relation từ raw LightRAG output ──────────────
def _parse_graph(items: list[dict]) -> tuple[str, str]:
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


# ── Prompt template sử gia trung lập ─────────────────────────────────────────
_PROMPT_TEMPLATE = """\
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


class VietnamHistoryQueryEngine:
    """Engine chatbot sử gia: hybrid retrieval + knowledge graph + LLM."""

    def __init__(self, top_k: int = 4, limit: int = 40):
        _logger.info("Khởi tạo VietnamHistoryQueryEngine (top_k=%d, limit=%d)", top_k, limit)

        self._top_k = top_k
        self._limit = limit

        self.api_key = _require_gemini_key()
        self.llm_model_name = _resolve_gemini_model_name()
        _logger.info("LLM model=%s", self.llm_model_name)

        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.dense_model = E5EmbeddingModel(E5EmbeddingConfig(prompt_name=E5_QUERY_PROMPT_NAME))
        self.sparse_model = SparseTextEmbedding("Qdrant/bm25")

        self.parent_store = self._load_parent_store()

        self.llm = _build_gemini_llm_func(
            gemini_key=self.api_key,
            gemini_model_name=self.llm_model_name,
            requests_per_minute=200,
            max_concurrency=4,
            transient_max_retries=3,
        )

        async def _embed_func(texts):
            return await asyncio.to_thread(self.dense_model.embed, texts)

        self.rag = LightRAG(
            working_dir=str(LIGHTRAG_WORKSPACE),
            llm_model_func=self.llm,
            embedding_func=EmbeddingFunc(
                embedding_dim=E5_EMBEDDING_DIM,
                max_token_size=E5_MAX_LENGTH,
                func=_embed_func,
            ),
        )

        self._rag_ready = False
        self._lock = asyncio.Lock()
        self._warmup_task: asyncio.Task | None = None

    def _load_parent_store(self) -> dict[str, str]:
        try:
            with open(PARENT_DOCSTORE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            _logger.warning(
                "Không tìm thấy %s. Retriever sẽ fallback về child chunk text — "
                "chất lượng truy hồi bị giảm. Hãy chạy pipeline index trước.",
                PARENT_DOCSTORE_PATH,
            )
            return {}
        except Exception as exc:
            _logger.warning("Không thể đọc parent_docs.json: %s", exc)
            return {}

    async def _init_rag(self) -> None:
        # Khởi tạo LightRAG lần đầu với khoá mutex để tránh khởi tạo song song
        if self._rag_ready:
            return
        async with self._lock:
            if not self._rag_ready:
                await self.rag.initialize_storages()
                self._rag_ready = True

    async def start(self) -> None:
        """Kick LightRAG warm-up chạy nền ngay sau khi engine khởi tạo xong."""
        self._warmup_task = asyncio.create_task(self._init_rag())
        self._warmup_task.add_done_callback(
            lambda t: (
                _logger.info("LightRAG warm-up hoàn tất.")
                if not t.cancelled() and t.exception() is None
                else _logger.warning("LightRAG warm-up thất bại: %s", t.exception())
            )
        )

    async def get_vector(self, query: str, top_k: int | None = None, limit: int | None = None) -> dict[str, Any]:
        return await get_vector(
            query=query,
            top_k=top_k if top_k is not None else self._top_k,
            limit=limit if limit is not None else self._limit,
            qdrant=self.qdrant,
            dense_model=self.dense_model,
            sparse_model=self.sparse_model,
            parent_store=self.parent_store,
        )

    async def _retrieve_decomposed(
        self, sub_queries: list[str], top_k_each: int = 3
    ) -> dict[str, Any]:
        """Retrieve song song theo từng sub-query triều đại, merge + dedup kết quả."""
        tasks = [self.get_vector(q, top_k=top_k_each) for q in sub_queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        seen_ids: set[str] = set()
        merged: list[dict] = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                _logger.warning("Sub-query '%s' thất bại: %s", sub_queries[i], res)
                continue
            for item in res.get("items", []):
                pid = item.get("parent_id") or item.get("id") or item.get("text", "")[:80]
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    merged.append(item)

        merged.sort(key=lambda x: x.get("score", 0), reverse=True)
        _logger.info("[decompose] merged %d unique items từ %d sub-queries", len(merged), len(sub_queries))
        return {"items": merged}

    async def get_graph(self, query: str, top_k: int = 10) -> dict[str, Any]:
        """Truy vấn đồ thị tri thức LightRAG."""
        await self._init_rag()

        try:
            raw = await self.rag.aquery(
                query,
                param=QueryParam(mode="local", only_need_context=True, top_k=top_k),
            )
        except Exception as exc:
            _logger.warning("LightRAG graph query thất bại: %s", exc, exc_info=True)
            return {"items": [], "error": str(exc)}

        text = _coerce_text(raw)
        return {"items": [{"text": b} for b in _split_blocks(text)]}

    async def ask_with_sources(self, question: str, history: str = "") -> dict[str, Any]:
        """Pipeline sử gia: query rewrite → vector + graph song song → prompt → LLM → trả lời kèm nguồn."""
        t0 = time.monotonic()
        _logger.info("[chatbot] Pipeline bắt đầu: %s", question[:80])

        retrieval_query = _rewrite_query(question)
        if retrieval_query != question:
            _logger.info("[rewrite] '%s' → '%s'", question[:60], retrieval_query[:60])

        is_broad = _is_broad_query(question) or _is_broad_query(retrieval_query)
        vec_top_k = _BROAD_TOP_K if is_broad else self._top_k
        graph_top_k = _BROAD_GRAPH_TOP_K if is_broad else 10

        if is_broad:
            sub_queries = _decompose_broad_query(retrieval_query)
            _logger.info("[decompose] %d sub-queries", len(sub_queries))
            vector_bundle, graph_bundle = await asyncio.gather(
                self._retrieve_decomposed(sub_queries, top_k_each=3),
                self.get_graph(retrieval_query, top_k=graph_top_k),
            )
        else:
            vector_bundle, graph_bundle = await asyncio.gather(
                self.get_vector(retrieval_query, top_k=vec_top_k),
                self.get_graph(retrieval_query, top_k=graph_top_k),
            )

        t1 = time.monotonic()
        vector_items = vector_bundle["items"]
        _logger.info(
            "[chatbot] retrieval %.2fs — vector=%d, graph=%d blocks",
            t1 - t0, len(vector_items), len(graph_bundle["items"]),
        )

        if not vector_items and not graph_bundle["items"]:
            return {
                "answer": "Tôi chưa tìm thấy tài liệu lịch sử chính xác về vấn đề này.",
                "sources": [],
                "verification": "Không tìm được nguồn vector phù hợp trong chỉ mục hiện tại.",
            }

        vector_context = _format_context_items(vector_items)
        entities, relations = _parse_graph(graph_bundle["items"])
        sources = _build_source_payload(vector_items)

        history_block = (
            f"▌LỊCH SỬ HỘI THOẠI (ngữ cảnh từ các lượt trước)\n{history}\n\n"
            if history else ""
        )
        prompt = _PROMPT_TEMPLATE.format(
            entities=entities,
            relations=relations,
            vector_context=vector_context,
            history_block=history_block,
            question=question,
        )

        try:
            answer = await self.llm(prompt)
        except Exception as exc:
            _logger.error("LLM thất bại: %s\n%s", exc, traceback.format_exc())
            raise

        t2 = time.monotonic()
        _logger.info("[chatbot] llm %.2fs | total %.2fs", t2 - t1, t2 - t0)

        graph_note = (
            f", {len(graph_bundle['items'])} blocks đồ thị làm giàu ngữ cảnh"
            if graph_bundle["items"] else ""
        )
        return {
            "answer": answer,
            "sources": sources,
            "verification": (
                f"Trả lời dựa trên {len(sources)} nguồn vector (trung tâm){graph_note}. "
                "Nhân vật: sử gia."
            ),
        }

    async def ask(self, question: str) -> str:
        result = await self.ask_with_sources(question)
        return result["answer"]

    async def ask_with_sources_stream(self, question: str, history: str = ""):
        """Streaming version: yields SSE event dicts token by token."""
        import openai as _openai

        t0 = time.monotonic()
        retrieval_query = _rewrite_query(question)
        is_broad = _is_broad_query(question) or _is_broad_query(retrieval_query)
        vec_top_k = _BROAD_TOP_K if is_broad else self._top_k
        graph_top_k = _BROAD_GRAPH_TOP_K if is_broad else 10

        if is_broad:
            vector_bundle, graph_bundle = await asyncio.gather(
                self._retrieve_decomposed(_decompose_broad_query(retrieval_query), top_k_each=3),
                self.get_graph(retrieval_query, top_k=graph_top_k),
            )
        else:
            vector_bundle, graph_bundle = await asyncio.gather(
                self.get_vector(retrieval_query, top_k=vec_top_k),
                self.get_graph(retrieval_query, top_k=graph_top_k),
            )

        vector_items = vector_bundle["items"]
        _logger.info("[stream] retrieval %.2fs — vector=%d", time.monotonic() - t0, len(vector_items))

        if not vector_items and not graph_bundle["items"]:
            yield {"type": "token", "text": "Tôi chưa tìm thấy tài liệu lịch sử chính xác về vấn đề này."}
            yield {"type": "done", "sources": []}
            return

        sources = _build_source_payload(vector_items)
        vector_context = _format_context_items(vector_items)
        entities, relations = _parse_graph(graph_bundle["items"])
        history_block = (
            f"▌LỊCH SỬ HỘI THOẠI (ngữ cảnh từ các lượt trước)\n{history}\n\n"
            if history else ""
        )
        prompt = _PROMPT_TEMPLATE.format(
            entities=entities,
            relations=relations,
            vector_context=vector_context,
            history_block=history_block,
            question=question,
        )

        client = _openai.AsyncOpenAI(api_key=self.api_key, base_url=GEMINI_OPENAI_BASE_URL)
        try:
            stream = await client.chat.completions.create(
                model=self.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield {"type": "token", "text": delta}
        except Exception as exc:
            _logger.error("[stream] LLM thất bại: %s", exc, exc_info=True)
            raise

        yield {"type": "done", "sources": sources}
