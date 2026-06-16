"""Engine persona chat: nhập vai nhân vật lịch sử khi trả lời câu hỏi.

Nhận VietnamHistoryQueryEngine qua dependency injection để tái dùng
vector search và knowledge graph mà không khởi tạo thêm model.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from services.chatbot.chatbot.engine import VietnamHistoryQueryEngine

from .persona_config import PersonaConfig, check_temporal_guardrail
from src.prompts.prompt_templates import (
    PERSONA_PROMPT as _PERSONA_PROMPT_TEMPLATE,
    rewrite_query as _rewrite_query,
    is_broad_query as _is_broad_query,
    decompose_broad_query as _decompose_broad_query,
    parse_graph as _parse_graph,
    build_retrieval_query as _build_retrieval_query,
    BROAD_TOP_K as _BROAD_TOP_K,
    BROAD_GRAPH_TOP_K as _BROAD_GRAPH_TOP_K,
)
from src.utils.helpers import (
    build_source_payload as _build_source_payload,
    format_context_items as _format_context_items,
)

_logger = logging.getLogger(__name__)


class PersonaChatEngine:
    """Engine xử lý câu hỏi theo nhân vật lịch sử cụ thể.

    Wrap VietnamHistoryQueryEngine để tái dùng toàn bộ retrieval pipeline,
    chỉ thay đổi phần prompt và guardrail theo persona.
    """

    def __init__(self, base_engine: "VietnamHistoryQueryEngine") -> None:
        self._engine = base_engine

    async def ask_with_sources(
        self,
        question: str,
        persona: PersonaConfig,
        history: str = "",
        turns: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Pipeline persona: guardrail → retrieval (tái dùng engine) → persona prompt → LLM."""
        # Bước 0: Time-Bound Guardrail
        blocked_reply = check_temporal_guardrail(question, persona)
        if blocked_reply:
            _logger.info(
                "[guardrail] Câu hỏi bị chặn cho persona '%s': %s",
                persona.slug, question[:80],
            )
            return {
                "answer": blocked_reply,
                "sources": [],
                "verification": (
                    f"Câu hỏi vượt ngoài mốc thời gian của "
                    f"{persona.display_name} ({persona.knowledge_cutoff_year})."
                ),
            }

        # Bước 1: Retrieval — tái dùng hoàn toàn từ base engine
        t0 = time.monotonic()
        turns = turns or []

        retrieval_base, _topic_shifted = _build_retrieval_query(question, turns)
        retrieval_query = _rewrite_query(retrieval_base)

        is_broad = _is_broad_query(question) or _is_broad_query(retrieval_query)
        vec_top_k = _BROAD_TOP_K if is_broad else self._engine._top_k
        graph_top_k = _BROAD_GRAPH_TOP_K if is_broad else 10

        if is_broad:
            sub_queries = _decompose_broad_query(retrieval_query)
            vector_bundle, graph_bundle = await asyncio.gather(
                self._engine._retrieve_decomposed(sub_queries, top_k_each=3),
                self._engine.get_graph(retrieval_query, top_k=graph_top_k),
            )
        else:
            vector_bundle, graph_bundle = await asyncio.gather(
                self._engine.get_vector(retrieval_query, top_k=vec_top_k),
                self._engine.get_graph(retrieval_query, top_k=graph_top_k),
            )

        t1 = time.monotonic()
        vector_items = vector_bundle["items"]
        _logger.info(
            "[persona:%s] retrieval %.2fs — vector=%d, graph=%d blocks",
            persona.slug, t1 - t0, len(vector_items), len(graph_bundle["items"]),
        )

        if not vector_items and not graph_bundle["items"]:
            no_data_reply = (
                f"Ta chưa tìm thấy sử liệu phù hợp với câu hỏi này trong kho tài liệu hiện tại. "
                f"Hãy thử đặt câu hỏi theo hướng khác về thời kỳ "
                f"{persona.knowledge_start_year}–{persona.knowledge_cutoff_year}."
            )
            return {
                "answer": no_data_reply,
                "sources": [],
                "verification": "Không tìm được nguồn vector phù hợp trong chỉ mục hiện tại.",
            }

        vector_context = _format_context_items(vector_items)
        entities, relations = _parse_graph(graph_bundle["items"])
        sources = _build_source_payload(vector_items)

        # Bước 2: Persona prompt
        history_block = (
            f"▌LỊCH SỬ HỘI THOẠI (ngữ cảnh từ các lượt trước)\n{history}\n\n"
            if history else ""
        )
        prompt = _PERSONA_PROMPT_TEMPLATE.format(
            system_prompt=persona.system_prompt,
            display_name=persona.display_name,
            knowledge_cutoff_year=persona.knowledge_cutoff_year,
            entities=entities,
            relations=relations,
            vector_context=vector_context,
            history_block=history_block,
            question=question,
        )

        answer = await self._engine.llm(prompt)
        t2 = time.monotonic()

        _logger.info(
            "[persona:%s] llm %.2fs | total %.2fs",
            persona.slug, t2 - t1, t2 - t0,
        )

        graph_note = (
            f", {len(graph_bundle['items'])} blocks đồ thị làm giàu ngữ cảnh"
            if graph_bundle["items"] else ""
        )
        verification = (
            f"Trả lời dựa trên {len(sources)} nguồn vector (trung tâm){graph_note}. "
            f"Nhân vật: {persona.display_name}."
        )
        graph_texts = [
            b["text"] for b in graph_bundle["items"]
            if isinstance(b.get("text"), str) and b["text"].strip()
        ]
        return {
            "answer": answer,
            "sources": sources,
            "verification": verification,
            "contexts": [item["text"] for item in vector_items if item.get("text")] + graph_texts,
        }

    async def ask(self, question: str, persona: PersonaConfig) -> str:
        result = await self.ask_with_sources(question, persona)
        return result["answer"]
