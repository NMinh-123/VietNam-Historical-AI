"""Pipeline ingest: điều phối Qdrant và LightRAG (có thể chạy riêng lẻ hoặc song song)."""

from __future__ import annotations

import json

from fastembed import SparseTextEmbedding
from langchain_core.documents import Document
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
    _resolve_qdrant_batch_size,
    _resolve_resume_existing_queue,
)
from .ingest_support import _apply_test_mode_subset, _validate_paths
from .lightrag_index import build_lightrag_instance, ingest_to_lightrag
from .providers import _build_gemini_llm_func
from .qdrant_index import ingest_to_qdrant


def _load_and_chunk_docs(
    dense_model: E5EmbeddingModel,
    test_mode: bool,
    parent_limit: int,
) -> tuple[list, list, dict]:
    """Tải PDF → làm sạch → phân mảnh parent/child. Trả về (parent_docs, child_chunks, parent_store)."""
    docs = load_pdfs_from_folder(str(RAW_DATA_PATH))
    if not docs:
        raise RuntimeError("Lỗi: Không tìm thấy file PDF trong thư mục dữ liệu.")
    docs = clean_documents(docs)

    child_chunks, parent_store, parent_docs = build_parent_child_chunks(
        docs, embed_fn=dense_model.embed
    )
    if not child_chunks or not parent_docs:
        raise RuntimeError("Lỗi: Không tạo được chunks từ tài liệu.")

    if test_mode:
        print(f"ĐANG CHẠY CHẾ ĐỘ TEST (Chỉ lấy {parent_limit} Parent đầu tiên)")
        parent_docs, child_chunks, parent_store = _apply_test_mode_subset(
            parent_docs=parent_docs,
            child_chunks=child_chunks,
            parent_store=parent_store,
            parent_limit=parent_limit,
        )

    return parent_docs, child_chunks, parent_store


def _load_parent_docs_from_disk() -> list[Document]:
    """Đọc lại parent docs đã lưu từ parent_docs.json để LightRAG không cần load PDF lại."""
    if not PARENT_DOCSTORE_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {PARENT_DOCSTORE_PATH}. "
            "Hãy chạy run_qdrant_index.py trước để tạo file này."
        )
    with open(PARENT_DOCSTORE_PATH, "r", encoding="utf-8") as f:
        parent_store: dict[str, str] = json.load(f)

    docs = [
        Document(
            page_content=text,
            metadata={"doc_id": parent_id, "parent_id": parent_id},
        )
        for parent_id, text in parent_store.items()
    ]
    print(f"Đọc lại {len(docs)} parent docs từ {PARENT_DOCSTORE_PATH}")
    return docs


async def qdrant_ingest(
    test_mode: bool = True,
    parent_limit: int = 10,
    recreate_collection: bool = True,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: str = COLLECTION_NAME,
    qdrant_batch_size: int | None = None,
) -> None:
    """Nạp toàn bộ dữ liệu vào Qdrant vector database."""
    print("=" * 60)
    print("PIPELINE QDRANT: NẠP DỮ LIỆU VECTOR")
    print("=" * 60)
    _validate_paths()

    resolved_qdrant_batch_size = _resolve_qdrant_batch_size(qdrant_batch_size)

    print(f"Đang tải {DENSE_MODEL_NAME}...")
    dense_model = E5EmbeddingModel(
        E5EmbeddingConfig(prompt_name=E5_PASSAGE_PROMPT_NAME, batch_size=32)
    )
    sparse_model = SparseTextEmbedding(SPARSE_MODEL_NAME)

    parent_docs, child_chunks, parent_store = _load_and_chunk_docs(
        dense_model, test_mode, parent_limit
    )

    save_parent_documents(parent_store, str(PARENT_DOCSTORE_PATH))
    save_documents(child_chunks, str(CHILD_DOCSTORE_PATH))
    print(f"Đã lưu {len(parent_docs)} parent docs → {PARENT_DOCSTORE_PATH}")
    print(f"Đã lưu {len(child_chunks)} child chunks → {CHILD_DOCSTORE_PATH}")

    print(f"\nĐANG NẠP {len(child_chunks)} CHILD CHUNKS VÀO QDRANT...")
    await ingest_to_qdrant(
        child_chunks=child_chunks,
        qdrant_client=QdrantClient(host=qdrant_host, port=qdrant_port),
        collection_name=collection_name,
        dense_model=dense_model,
        sparse_model=sparse_model,
        batch_size=resolved_qdrant_batch_size,
        recreate_collection=recreate_collection,
    )

    print("\nHOÀN TẤT NẠP DỮ LIỆU VECTOR VÀO QDRANT!")


async def lightrag_ingest(
    test_mode: bool = True,
    parent_limit: int = 10,
    gemini_key: str | None = None,
    gemini_model_name: str | None = None,
    gemini_requests_per_minute: int | None = None,
    gemini_max_concurrency: int | None = None,
    gemini_transient_max_retries: int | None = None,
    lightrag_batch_size: int | None = None,
    lightrag_max_parallel_insert: int | None = None,
    resume_existing_queue: bool | None = None,
    use_saved_docs: bool = True,
) -> None:
    """Nạp dữ liệu vào LightRAG để xây dựng knowledge graph."""
    print("=" * 60)
    print("PIPELINE LIGHTRAG: NẠP DỮ LIỆU KNOWLEDGE GRAPH")
    print("=" * 60)
    _validate_paths()

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
        f"LLM: model={resolved_gemini_model_name}, "
        f"rpm_limit={resolved_gemini_rpm_limit}, "
        f"max_concurrency={resolved_gemini_max_concurrency}, "
        f"base_url={GEMINI_OPENAI_BASE_URL}"
    )

    gemini_llm_func = _build_gemini_llm_func(
        resolved_gemini_key,
        resolved_gemini_model_name,
        resolved_gemini_rpm_limit,
        resolved_gemini_max_concurrency,
        resolved_gemini_transient_max_retries,
    )

    print(f"Đang tải {DENSE_MODEL_NAME}...")
    dense_model = E5EmbeddingModel(
        E5EmbeddingConfig(prompt_name=E5_PASSAGE_PROMPT_NAME, batch_size=32)
    )

    if use_saved_docs:
        parent_docs = _load_parent_docs_from_disk()
        if test_mode:
            parent_docs = parent_docs[:parent_limit]
            print(f"TEST MODE: Giới hạn còn {len(parent_docs)} parent docs")
    else:
        parent_docs, _child_chunks, _parent_store = _load_and_chunk_docs(
            dense_model, test_mode, parent_limit
        )

    print(f"\nĐANG NẠP {len(parent_docs)} PARENT DOCS VÀO LIGHTRAG...")
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

    print("\nHOÀN TẤT NẠP DỮ LIỆU KNOWLEDGE GRAPH VÀO LIGHTRAG!")


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
    qdrant_batch_size: int | None = None,
    lightrag_batch_size: int | None = None,
    lightrag_max_parallel_insert: int | None = None,
    resume_existing_queue: bool | None = None,
) -> None:
    """Chạy cả hai pipeline Qdrant và LightRAG liên tiếp."""
    print("BẮT ĐẦU PIPELINE NẠP KÉP (VECTOR + KNOWLEDGE GRAPH)...")

    await qdrant_ingest(
        test_mode=test_mode,
        parent_limit=parent_limit,
        recreate_collection=recreate_collection,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        collection_name=collection_name,
        qdrant_batch_size=qdrant_batch_size,
    )

    await lightrag_ingest(
        test_mode=test_mode,
        parent_limit=parent_limit,
        gemini_key=gemini_key,
        gemini_model_name=gemini_model_name,
        gemini_requests_per_minute=gemini_requests_per_minute,
        gemini_max_concurrency=gemini_max_concurrency,
        gemini_transient_max_retries=gemini_transient_max_retries,
        lightrag_batch_size=lightrag_batch_size,
        lightrag_max_parallel_insert=lightrag_max_parallel_insert,
        resume_existing_queue=resume_existing_queue,
        use_saved_docs=True,
    )

    print("\nHOÀN TẤT NẠP KÉP THÀNH CÔNG!")


__all__ = ["hybrid_ingest", "qdrant_ingest", "lightrag_ingest"]
