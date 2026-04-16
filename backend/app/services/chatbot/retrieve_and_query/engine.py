"""Engine truy vấn lịch sử Việt Nam: điều phối vector search + knowledge graph + LLM."""

from __future__ import annotations

import asyncio
import json
import logging
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
from services.chatbot.index import (
    LIGHTRAG_WORKSPACE,
    PARENT_DOCSTORE_PATH,
    EmbeddingFunc,
    LightRAG,
    _build_gemini_llm_func,
    _require_gemini_key,
    _resolve_gemini_model_name,
)
from .context_builder import (
    _build_source_payload,
    _coerce_text,
    _format_context_items,
    _format_graph_context_items,
    _split_blocks,
)
from .retriever import get_vector

_PROMPT_TEMPLATE = """\
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


class VietnamHistoryQueryEngine:
    def __init__(self):
        print("Initializing Query Engine...")

        self.api_key = _require_gemini_key()
        self.llm_model_name = _resolve_gemini_model_name()

        self.qdrant = QdrantClient(host="localhost", port=6333)
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

        async def embed_func(texts):
            return await asyncio.to_thread(self.dense_model.embed, texts)

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
        self._warmup_task: asyncio.Task | None = None

    def _load_parent_store(self) -> dict[str, str]:
        # Tải kho văn bản cha từ file JSON; trả dict rỗng nếu file chưa tồn tại
        try:
            with open(PARENT_DOCSTORE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(
                f"[WARNING] Không tìm thấy {PARENT_DOCSTORE_PATH}. "
                "Retriever sẽ fallback về child chunk text — chất lượng truy hồi bị giảm. "
                "Hãy chạy pipeline index trước."
            )
            return {}
        except Exception as exc:
            print(f"[WARNING] Không thể đọc parent_docs.json: {exc}")
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
        """Kick LightRAG warm-up chạy nền ngay sau khi engine khởi tạo xong.

        Gọi một lần sau khi tạo engine, trước khi vào vòng lặp chính.
        Đến khi user hỏi câu đầu tiên, LightRAG thường đã sẵn sàng.
        """
        self._warmup_task = asyncio.create_task(self._init_rag())
        self._warmup_task.add_done_callback(
            lambda t: print("[OK] LightRAG warm-up hoàn tất.")
            if not t.cancelled() and t.exception() is None
            else print(f"[WARNING] LightRAG warm-up thất bại: {t.exception()}")
        )

    async def get_vector(self, query: str, top_k: int = 4, limit: int = 40) -> dict[str, Any]:
        # Truy hồi hybrid vector từ Qdrant, tái xếp hạng và đa dạng hoá kết quả
        return await get_vector(
            query=query,
            top_k=top_k,
            limit=limit,
            qdrant=self.qdrant,
            dense_model=self.dense_model,
            sparse_model=self.sparse_model,
            parent_store=self.parent_store,
        )

    async def get_graph(
        self, query: str, vector_items: list[dict[str, Any]], top_k: int = 10
    ) -> dict[str, Any]:
        # Truy vấn đồ thị tri thức LightRAG; dùng kết quả vector làm seed để hướng dẫn tìm kiếm
        await self._init_rag()

        seed = " ".join([item["text"][:200] for item in vector_items])
        guided_query = f"{query}\n{seed}"

        try:
            raw = await self.rag.aquery(
                guided_query,
                param=QueryParam(
                    mode="local",
                    only_need_context=True,
                    top_k=top_k,
                ),
            )
        except Exception as exc:
            _logger.warning("LightRAG graph query thất bại: %s", exc, exc_info=True)
            return {"items": []}

        text = _coerce_text(raw)
        return {"items": [{"text": b} for b in _split_blocks(text)]}

    async def ask_with_sources(self, question: str) -> dict[str, Any]:
        # Pipeline chính: truy hồi vector + đồ thị → tổng hợp prompt → gọi LLM → trả câu trả lời kèm nguồn
        print("\nRunning pipeline...")

        vector_bundle = await self.get_vector(question)
        vector_items = vector_bundle["items"]

        graph_bundle = await self.get_graph(question, vector_items)

        if not vector_items and not graph_bundle["items"]:
            return {
                "answer": "Tôi chưa tìm thấy tài liệu lịch sử chính xác về vấn đề này.",
                "sources": [],
                "verification": "Không tìm được nguồn vector phù hợp trong chỉ mục hiện tại.",
            }

        vector_context = _format_context_items(vector_items)
        graph_context = _format_graph_context_items(graph_bundle["items"])
        sources = _build_source_payload(vector_items)

        prompt = _PROMPT_TEMPLATE.format(
            vector_context=vector_context,
            graph_context=graph_context,
            question=question,
        )

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

    async def ask(self, question: str) -> str:
        # Gọi ask_with_sources và chỉ trả về phần câu trả lời văn bản
        result = await self.ask_with_sources(question)
        return result["answer"]


async def main() -> None:
    # Vòng lặp REPL đơn giản để kiểm thử engine từ dòng lệnh; gõ "q" hoặc "exit" để thoát
    engine = VietnamHistoryQueryEngine()

    while True:
        q = input("\n> ")
        if q.lower() in ["q", "exit"]:
            break

        ans = await engine.ask(q)
        print("\n", ans)


if __name__ == "__main__":
    asyncio.run(main())
