"""Engine chatbot sử gia: điều phối vector search + knowledge graph + LLM.

Chỉ xử lý chế độ sử gia trung lập (không persona).
Persona mode được xử lý riêng bởi PersonaChatEngine trong persona_chat/.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from typing import Any

_logger = logging.getLogger(__name__)

from fastembed import SparseTextEmbedding
from lightrag import QueryParam
from qdrant_client import QdrantClient

from src.embeddings.embedder import (
    E5_EMBEDDING_DIM,
    E5_MAX_LENGTH,
    E5_QUERY_PROMPT_NAME,
    E5EmbeddingConfig,
    E5EmbeddingModel,
)
from src.llm.llm_client import (
    LLM_BASE_URL as GEMINI_OPENAI_BASE_URL,
    build_llm_func as _build_gemini_llm_func,
    require_api_key as _require_gemini_key,
    resolve_model_name as _resolve_gemini_model_name,
)
from services.chatbot.index_and_retrieve import (
    LIGHTRAG_WORKSPACE,
    PARENT_COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    EmbeddingFunc,
    LightRAG,
)
from src.utils.helpers import (
    build_source_payload as _build_source_payload,
    coerce_text as _coerce_text,
    format_context_items as _format_context_items,
    split_blocks as _split_blocks,
)
from src.retrieval.retriever import retrieve as get_vector
from src.prompts.prompt_templates import (
    HISTORIAN_PROMPT as _PROMPT_TEMPLATE,
    rewrite_query as _rewrite_query,
    is_broad_query as _is_broad_query,
    decompose_broad_query as _decompose_broad_query,
    parse_graph as _parse_graph,
    build_retrieval_query as _build_retrieval_query,
    BROAD_TOP_K as _BROAD_TOP_K,
    BROAD_GRAPH_TOP_K as _BROAD_GRAPH_TOP_K,
)


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

        self.llm = _build_gemini_llm_func(
            api_key=self.api_key,
            model_name=self.llm_model_name,
            requests_per_minute=200,
            max_concurrency=4,
            max_retries=3,
        )
        import openai as _openai
        self._stream_client = _openai.AsyncOpenAI(api_key=self.api_key, base_url=GEMINI_OPENAI_BASE_URL)
        self._stream_semaphore = asyncio.Semaphore(4)

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
            parent_collection=PARENT_COLLECTION_NAME,
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

    async def ask_with_sources(
        self, question: str, history: str = "", turns: list[dict] | None = None
    ) -> dict[str, Any]:
        """Pipeline sử gia: standalone Q → entity tracking → window retrieval → prompt → LLM."""
        t0 = time.monotonic()
        _logger.info("[chatbot] Pipeline bắt đầu: %s", question[:80])
        turns = turns or []

        # 0 LLM calls — pure Python: topic shift + window context
        retrieval_base, _topic_shifted = _build_retrieval_query(question, turns)
        retrieval_query = _rewrite_query(retrieval_base)

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
                "contexts": [],
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
        # contexts gồm cả vector chunks lẫn graph blocks để RAGAS đánh giá đúng
        # vì câu trả lời có thể dựa chủ yếu vào graph khi vector search miss
        graph_texts = [
            b["text"] for b in graph_bundle["items"]
            if isinstance(b.get("text"), str) and b["text"].strip()
        ]
        all_contexts = [item["text"] for item in vector_items if item.get("text")] + graph_texts

        return {
            "answer": answer,
            "sources": sources,
            "contexts": all_contexts,
            "verification": (
                f"Trả lời dựa trên {len(sources)} nguồn vector (trung tâm){graph_note}. "
                "Nhân vật: sử gia."
            ),
        }

    async def ask(self, question: str) -> str:
        result = await self.ask_with_sources(question)
        return result["answer"]

    async def ask_with_sources_stream(
        self, question: str, history: str = "", turns: list[dict] | None = None
    ):
        """Streaming version: yields SSE event dicts token by token."""
        t0 = time.monotonic()
        turns = turns or []

        # 0 LLM calls — pure Python: topic shift + window context
        retrieval_base, _topic_shifted = _build_retrieval_query(question, turns)
        retrieval_query = _rewrite_query(retrieval_base)

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

        try:
            async with self._stream_semaphore:
                stream = await self._stream_client.chat.completions.create(
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
