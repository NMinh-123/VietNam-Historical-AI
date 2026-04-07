from __future__ import annotations

import asyncio
import os
import sys
import uuid
from importlib import import_module
from pathlib import Path

import numpy as np
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
# NHÓM 1: IMPORT DỮ LIỆU & CÔNG CỤ
CURRENT_DIR = Path(__file__).resolve().parent
APP_DIR = CURRENT_DIR.parent
DATA_DIR = APP_DIR / "data"
SERVICE_DIR = CURRENT_DIR

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from data.process_data.clean_data import clean_documents
from data.process_data.e5_embeddings import (
    E5_EMBEDDING_DIM,
    E5_EMBEDDING_MODEL_NAME,
    E5_MAX_LENGTH,
    E5_PASSAGE_PROMPT_NAME,
    E5EmbeddingConfig,
    E5EmbeddingModel,
)
from data.process_data.load_data import load_pdfs_from_folder
from data.process_data.splitter import (
    build_parent_child_chunks,
    save_documents,
    save_parent_documents,
)


def _import_external_lightrag():
    """Import package LightRAG ngoài site-packages, tránh đụng chính file này."""
    original_sys_path = list(sys.path)
    try:
        sys.path[:] = [
            path
            for path in sys.path
            if Path(path or ".").resolve() != SERVICE_DIR
        ]
        lightrag_module = import_module("lightrag")
        llm_openai_module = import_module("lightrag.llm.openai")
        utils_module = import_module("lightrag.utils")
        return (
            lightrag_module.LightRAG,
            llm_openai_module.openai_complete_if_cache,
            utils_module.EmbeddingFunc,
        )
    finally:
        sys.path[:] = original_sys_path


LightRAG, openai_complete_if_cache, EmbeddingFunc = _import_external_lightrag()

# NHÓM 2: CẤU HÌNH ĐƯỜNG DẪN & API
RAW_DATA_PATH = DATA_DIR / "raw_data"
QDRANT_DB_PATH = DATA_DIR / "qdrant_db"
LIGHTRAG_WORKSPACE = DATA_DIR / "lightrag_storage"
PARENT_DOCSTORE_PATH = DATA_DIR / "parent_docs.json"
CHILD_DOCSTORE_PATH = DATA_DIR / "child_docs.json"

COLLECTION_NAME = "vietnam_history_hybrid"
DENSE_MODEL_NAME = E5_EMBEDDING_MODEL_NAME
SPARSE_MODEL_NAME = "Qdrant/bm25"
DEFAULT_GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEPRECATED_GEMINI_MODEL_REPLACEMENTS = {
    "gemini-1.5-flash": "gemini-2.5-flash",
    "gemini-1.5-flash-001": "gemini-2.5-flash",
    "gemini-1.5-flash-002": "gemini-2.5-flash",
    "gemini-1.5-pro": "gemini-2.5-pro",
    "gemini-1.5-pro-001": "gemini-2.5-pro",
    "gemini-1.5-pro-002": "gemini-2.5-pro",
}
FREE_TIER_RPM_BY_MODEL_PREFIX = {
    "gemini-2.5-pro": 5,
    "gemini-2.5-flash": 10,
}
DEFAULT_GEMINI_MAX_CONCURRENCY = 1
DEFAULT_GEMINI_TRANSIENT_MAX_RETRIES = 4
GEMINI_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
GEMINI_TRANSIENT_RETRY_DELAYS_SECONDS = (15.0, 30.0, 60.0, 120.0)


def _read_env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _read_env_int(name: str) -> int | None:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return None
    return int(raw_value.strip())


def _require_gemini_key(gemini_key: str | None = None) -> str:
    key = gemini_key or os.getenv("GEMINI_KEY")
    if not key:
        raise ValueError("KHÔNG TÌM THẤY 'GEMINI_KEY' TRONG MÔI TRƯỜNG!")
    return key


def _resolve_gemini_model_name(gemini_model_name: str | None = None) -> str:
    requested_model = (
        gemini_model_name
        or os.getenv("GEMINI_MODEL_NAME")
        or DEFAULT_GEMINI_MODEL_NAME
    ).strip()
    resolved_model = DEPRECATED_GEMINI_MODEL_REPLACEMENTS.get(
        requested_model,
        requested_model,
    )
    if resolved_model != requested_model:
        print(
            f"Gemini model '{requested_model}' đã ngừng hỗ trợ. "
            f"Tự động chuyển sang '{resolved_model}'."
        )
    return resolved_model


def _resolve_gemini_rpm_limit(
    gemini_model_name: str,
    requests_per_minute: int | None = None,
) -> int:
    if requests_per_minute is not None:
        return requests_per_minute

    env_override = _read_env_int("GEMINI_RPM_LIMIT")
    if env_override is not None:
        return env_override

    for model_prefix, rpm_limit in FREE_TIER_RPM_BY_MODEL_PREFIX.items():
        if gemini_model_name.startswith(model_prefix):
            return rpm_limit

    return 10


def _resolve_gemini_max_concurrency(
    max_concurrency: int | None = None,
) -> int:
    if max_concurrency is not None:
        return max_concurrency

    env_override = _read_env_int("GEMINI_MAX_CONCURRENCY")
    if env_override is not None:
        return env_override

    return DEFAULT_GEMINI_MAX_CONCURRENCY


def _resolve_gemini_transient_max_retries(
    max_retries: int | None = None,
) -> int:
    if max_retries is not None:
        return max_retries

    env_override = _read_env_int("GEMINI_TRANSIENT_MAX_RETRIES")
    if env_override is not None:
        return env_override

    return DEFAULT_GEMINI_TRANSIENT_MAX_RETRIES


def _extract_exception_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return status_code

    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def _is_retryable_gemini_exception(exc: Exception) -> bool:
    status_code = _extract_exception_status_code(exc)
    if status_code in GEMINI_RETRYABLE_STATUS_CODES:
        return True

    error_text = str(exc).lower()
    retryable_markers = (
        "high demand",
        "unavailable",
        "temporarily unavailable",
        "try again later",
        "rate limit",
        "too many requests",
    )
    return any(marker in error_text for marker in retryable_markers)


class AsyncRequestRateLimiter:
    def __init__(self, max_requests: int, window_seconds: float = 60.0):
        if max_requests <= 0:
            raise ValueError("max_requests phải lớn hơn 0")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.min_interval_seconds = window_seconds / max_requests
        self._next_available_at = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        loop = asyncio.get_running_loop()
        async with self._lock:
            now = loop.time()
            scheduled_at = max(now, self._next_available_at)
            self._next_available_at = (
                scheduled_at + self.min_interval_seconds
            )

        sleep_for = scheduled_at - now
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)


def _validate_paths() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục raw_data tại: {RAW_DATA_PATH}"
        )

    PARENT_DOCSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIGHTRAG_WORKSPACE.mkdir(parents=True, exist_ok=True)
    QDRANT_DB_PATH.mkdir(parents=True, exist_ok=True)


def _build_gemini_llm_func(
    gemini_key: str,
    gemini_model_name: str,
    requests_per_minute: int,
    max_concurrency: int,
    transient_max_retries: int,
):
    request_limiter = AsyncRequestRateLimiter(
        max_requests=requests_per_minute,
        window_seconds=60.0,
    )
    request_semaphore = asyncio.Semaphore(max_concurrency)

    async def _gemini_llm_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        **kwargs,
    ):
        async with request_semaphore:
            for attempt_index in range(transient_max_retries):
                await request_limiter.acquire()
                try:
                    return await openai_complete_if_cache(
                        gemini_model_name,
                        prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages or [],
                        api_key=gemini_key,
                        base_url=GEMINI_OPENAI_BASE_URL,
                        **kwargs,
                    )
                except Exception as exc:
                    is_last_attempt = (
                        attempt_index == transient_max_retries - 1
                    )
                    if is_last_attempt or not _is_retryable_gemini_exception(exc):
                        raise

                    retry_delay_seconds = GEMINI_TRANSIENT_RETRY_DELAYS_SECONDS[
                        min(
                            attempt_index,
                            len(GEMINI_TRANSIENT_RETRY_DELAYS_SECONDS) - 1,
                        )
                    ]
                    status_code = _extract_exception_status_code(exc)
                    print(
                        "Gemini tạm quá tải hoặc chạm retryable error "
                        f"(status={status_code}, attempt={attempt_index + 1}/"
                        f"{transient_max_retries}). "
                        f"Chờ {retry_delay_seconds:.0f}s rồi thử lại..."
                    )
                    await asyncio.sleep(retry_delay_seconds)

    return _gemini_llm_func


def _apply_test_mode_subset(
    parent_docs: list,
    child_chunks: list,
    parent_store: dict,
    parent_limit: int,
) -> tuple[list, list, dict]:
    selected_parent_docs = parent_docs[:parent_limit]
    valid_parent_ids = {
        doc.metadata["doc_id"]
        for doc in selected_parent_docs
        if doc.metadata.get("doc_id")
    }
    selected_child_chunks = [
        chunk
        for chunk in child_chunks
        if chunk.metadata.get("parent_id") in valid_parent_ids
    ]
    selected_parent_store = {
        parent_id: parent_text
        for parent_id, parent_text in parent_store.items()
        if parent_id in valid_parent_ids
    }
    return selected_parent_docs, selected_child_chunks, selected_parent_store


def _prepare_qdrant_collection(
    client: QdrantClient,
    collection_name: str,
    recreate_collection: bool,
) -> None:
    if client.collection_exists(collection_name):
        if not recreate_collection:
            raise RuntimeError(
                f"Collection '{collection_name}' đã tồn tại. "
                "Nếu muốn build lại từ đầu, hãy truyền recreate_collection=True."
            )
        print(f"🧹 Xóa collection Qdrant cũ: {collection_name}")
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=E5_EMBEDDING_DIM,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )

# NHÓM 3: HÀM NẠP KÉP (DUAL INGESTION)

async def hybrid_ingest(
    test_mode: bool = True,
    parent_limit: int = 10,
    recreate_collection: bool = False,
    gemini_key: str | None = None,
    gemini_model_name: str | None = None,
    gemini_requests_per_minute: int | None = None,
    gemini_max_concurrency: int | None = None,
    gemini_transient_max_retries: int | None = None,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: str = COLLECTION_NAME,
) -> None:
    print("🚀 BẮT ĐẦU PIPELINE NẠP KÉP (VECTOR + KNOWLEDGE GRAPH)...")
    _validate_paths()
    resolved_gemini_key = _require_gemini_key(gemini_key)
    resolved_gemini_model_name = _resolve_gemini_model_name(gemini_model_name)
    resolved_gemini_rpm_limit = _resolve_gemini_rpm_limit(
        resolved_gemini_model_name,
        gemini_requests_per_minute,
    )
    resolved_gemini_max_concurrency = _resolve_gemini_max_concurrency(
        gemini_max_concurrency
    )
    resolved_gemini_transient_max_retries = (
        _resolve_gemini_transient_max_retries(
            gemini_transient_max_retries
        )
    )
    print(
        "LLM provider: Gemini (OpenAI-compatible), "
        f"model={resolved_gemini_model_name}, "
        f"rpm_limit={resolved_gemini_rpm_limit}, "
        f"max_concurrency={resolved_gemini_max_concurrency}, "
        f"retry_attempts={resolved_gemini_transient_max_retries}, "
        f"base_url={GEMINI_OPENAI_BASE_URL}"
    )
    gemini_llm_func = _build_gemini_llm_func(
        resolved_gemini_key,
        resolved_gemini_model_name,
        resolved_gemini_rpm_limit,
        resolved_gemini_max_concurrency,
        resolved_gemini_transient_max_retries,
    )

    # 1. Chuẩn bị dữ liệu
    docs = load_pdfs_from_folder(str(RAW_DATA_PATH))
    if not docs:
        print("Lỗi: Không tìm thấy file PDF.")
        return
    docs = clean_documents(docs)

    child_chunks, parent_store, parent_docs = build_parent_child_chunks(docs)
    if not child_chunks or not parent_docs:
        print("Lỗi: Không tạo được chunks.")
        return

    if test_mode:
        print(
            f"ĐANG CHẠY CHẾ ĐỘ TEST (Chỉ lấy {parent_limit} Parent đầu tiên)"
        )
        parent_docs, child_chunks, parent_store = _apply_test_mode_subset(
            parent_docs=parent_docs,
            child_chunks=child_chunks,
            parent_store=parent_store,
            parent_limit=parent_limit,
        )

    save_parent_documents(parent_store, str(PARENT_DOCSTORE_PATH))
    save_documents(child_chunks, str(CHILD_DOCSTORE_PATH))

    # Khởi tạo mô hình Embedding (Dùng chung cho cả Qdrant và LightRAG)
    print(f"Đang tải {DENSE_MODEL_NAME}...")
    dense_model = E5EmbeddingModel(
        E5EmbeddingConfig(
            prompt_name=E5_PASSAGE_PROMPT_NAME,
            batch_size=32,
        )
    )
    sparse_model = SparseTextEmbedding(SPARSE_MODEL_NAME)

    # NHÁNH 1: NẠP VÀO QDRANT
    # print(f"\n[1/2] ĐANG NẠP {len(child_chunks)} CHILD CHUNKS VÀO QDRANT...")
    # client = QdrantClient(host=qdrant_host, port=qdrant_port)
    # _prepare_qdrant_collection(client, collection_name, recreate_collection)

    # texts_to_embed = [doc.page_content for doc in child_chunks]
    # batch_size = 256
    # total_child = len(texts_to_embed)

    # for index in range(0, total_child, batch_size):
    #     batch_texts = texts_to_embed[index:index + batch_size]
    #     batch_docs = child_chunks[index:index + batch_size]

    #     batch_dense = dense_model.embed(batch_texts)
    #     batch_sparse = list(sparse_model.embed(batch_texts, parallel=0))

    #     points = []
    #     for batch_index in range(len(batch_texts)):
    #         points.append(
    #             models.PointStruct(
    #                 id=str(uuid.uuid4()),
    #                 vector={
    #                     "dense": batch_dense[batch_index].tolist(),
    #                     "sparse": models.SparseVector(
    #                         indices=batch_sparse[batch_index].indices.tolist(),
    #                         values=batch_sparse[batch_index].values.tolist(),
    #                     ),
    #                 },
    #                 payload={
    #                     "page_content": batch_docs[batch_index].page_content,
    #                     **batch_docs[batch_index].metadata,
    #                 },
    #             )
    #         )

    #     client.upsert(collection_name=collection_name, points=points)
    #     print(f"Qdrant: {min(index + batch_size, total_child)}/{total_child}")

    # NHÁNH 2: NẠP VÀO LIGHTRAG
    print(f"\n[2/2] ĐANG NẠP {len(parent_docs)} PARENT CHUNKS VÀO LIGHTRAG...")

    async def custom_fastembed(texts: list[str]) -> np.ndarray:
        return dense_model.embed(texts)
    rag = LightRAG(
        working_dir=str(LIGHTRAG_WORKSPACE),
        llm_model_func=gemini_llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=E5_EMBEDDING_DIM,
            max_token_size=E5_MAX_LENGTH,
            func=custom_fastembed,
        ),
        chunk_token_size=100000,
        chunk_overlap_token_size=0,
        max_parallel_insert=resolved_gemini_max_concurrency,
    )
    await rag.initialize_storages()

    parent_texts = [doc.page_content for doc in parent_docs]
    await rag.ainsert(parent_texts)

    print("HOÀN TẤT NẠP KÉP THÀNH CÔNG!")


if __name__ == "__main__":
    asyncio.run(
        hybrid_ingest(
            test_mode=_read_env_flag("LIGHTRAG_TEST_MODE", default=False),
            recreate_collection=_read_env_flag(
                "LIGHTRAG_RECREATE_QDRANT",
                default=False,
            ),
        )
    )
