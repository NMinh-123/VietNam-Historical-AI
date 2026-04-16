"""Pipeline ingest chính: điều phối Qdrant và LightRAG."""

from __future__ import annotations

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient

from data.process_data.clean_data import clean_documents
from data.process_data.e5_embeddings import (
    E5EmbeddingConfig,
    E5EmbeddingModel,
    E5_PASSAGE_PROMPT_NAME,
)
from data.process_data.load_data import load_pdfs_from_folder
from data.process_data.splitter import (
    build_parent_child_chunks,
    save_documents,
    save_parent_documents,
)
from .config import (
    CHILD_DOCSTORE_PATH,
    COLLECTION_NAME,
    DENSE_MODEL_NAME,
    GEMINI_OPENAI_BASE_URL,
    LIGHTRAG_WORKSPACE,
    PARENT_DOCSTORE_PATH,
    RAW_DATA_PATH,
    SPARSE_MODEL_NAME,
    _require_gemini_key,
    _resolve_gemini_max_concurrency,
    _resolve_gemini_model_name,
    _resolve_gemini_rpm_limit,
    _resolve_gemini_transient_max_retries,
    _resolve_lightrag_batch_size,
    _resolve_lightrag_max_parallel_insert,
    _resolve_resume_existing_queue,
)
from .ingest_support import _apply_test_mode_subset, _validate_paths
from ..lightrag_index import build_lightrag_instance, ingest_to_lightrag
from .providers import _build_gemini_llm_func
from ..qdrant_index import ingest_to_qdrant


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
    lightrag_batch_size: int | None = None,
    lightrag_max_parallel_insert: int | None = None,
    resume_existing_queue: bool | None = None,
) -> None:
    """Điều phối toàn bộ luồng ingest từ PDF thô sang Qdrant và LightRAG."""
    print("🚀 BẮT ĐẦU PIPELINE NẠP KÉP (VECTOR + KNOWLEDGE GRAPH)...")
    _validate_paths()

    # Giải quyết toàn bộ tham số cấu hình trước khi chạy
    resolved_gemini_key = _require_gemini_key(gemini_key)
    resolved_gemini_model_name = _resolve_gemini_model_name(gemini_model_name)
    resolved_gemini_rpm_limit = _resolve_gemini_rpm_limit(
        resolved_gemini_model_name,
        gemini_requests_per_minute,
    )
    resolved_lightrag_batch_size = _resolve_lightrag_batch_size(lightrag_batch_size)
    resolved_lightrag_max_parallel_insert = _resolve_lightrag_max_parallel_insert(
        lightrag_max_parallel_insert
    )
    resolved_resume_existing_queue = _resolve_resume_existing_queue(resume_existing_queue)
    resolved_gemini_max_concurrency = _resolve_gemini_max_concurrency(
        gemini_max_concurrency or resolved_lightrag_max_parallel_insert
    )
    resolved_gemini_transient_max_retries = _resolve_gemini_transient_max_retries(
        gemini_transient_max_retries
    )

    print(
        "LLM provider: ShopAIKey / OpenAI-compatible, "
        f"model={resolved_gemini_model_name}, "
        f"rpm_limit={resolved_gemini_rpm_limit}, "
        f"max_concurrency={resolved_gemini_max_concurrency}, "
        f"retry_attempts={resolved_gemini_transient_max_retries}, "
        f"batch_size={resolved_lightrag_batch_size}, "
        f"max_parallel_insert={resolved_lightrag_max_parallel_insert}, "
        f"base_url={GEMINI_OPENAI_BASE_URL}"
    )

    gemini_llm_func = _build_gemini_llm_func(
        resolved_gemini_key,
        resolved_gemini_model_name,
        resolved_gemini_rpm_limit,
        resolved_gemini_max_concurrency,
        resolved_gemini_transient_max_retries,
    )

    # Tải, làm sạch và phân mảnh tài liệu PDF
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
        print(f"ĐANG CHẠY CHẾ ĐỘ TEST (Chỉ lấy {parent_limit} Parent đầu tiên)")
        parent_docs, child_chunks, parent_store = _apply_test_mode_subset(
            parent_docs=parent_docs,
            child_chunks=child_chunks,
            parent_store=parent_store,
            parent_limit=parent_limit,
        )

    save_parent_documents(parent_store, str(PARENT_DOCSTORE_PATH))
    save_documents(child_chunks, str(CHILD_DOCSTORE_PATH))

    print(f"Đang tải {DENSE_MODEL_NAME}...")
    dense_model = E5EmbeddingModel(
        E5EmbeddingConfig(prompt_name=E5_PASSAGE_PROMPT_NAME, batch_size=32)
    )
    sparse_model = SparseTextEmbedding(SPARSE_MODEL_NAME)

    # Nhánh 1: Qdrant — nhúng child chunks vào vector database
    print(f"\n[1/2] ĐANG NẠP {len(child_chunks)} CHILD CHUNKS VÀO QDRANT...")
    await ingest_to_qdrant(
        child_chunks=child_chunks,
        qdrant_client=QdrantClient(host=qdrant_host, port=qdrant_port),
        collection_name=collection_name,
        dense_model=dense_model,
        sparse_model=sparse_model,
        recreate_collection=recreate_collection,
    )

    # Nhánh 2: LightRAG — xây dựng knowledge graph từ parent chunks
    print(f"\n[2/2] ĐANG NẠP {len(parent_docs)} PARENT CHUNKS VÀO LIGHTRAG...")
    rag = build_lightrag_instance(
        dense_model=dense_model,
        llm_func=gemini_llm_func,
        working_dir=str(LIGHTRAG_WORKSPACE),
        max_parallel_insert=resolved_lightrag_max_parallel_insert,
    )
    await ingest_to_lightrag(
        parent_docs=parent_docs,
        rag=rag,
        batch_size=resolved_lightrag_batch_size,
        resume_existing_queue=resolved_resume_existing_queue,
    )

    print("HOÀN TẤT NẠP KÉP THÀNH CÔNG!")


__all__ = ["hybrid_ingest"]
