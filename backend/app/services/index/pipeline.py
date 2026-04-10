"""Pipeline ingest chính cho Qdrant + LightRAG."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from data.process_data.clean_data import clean_documents
from data.process_data.e5_embeddings import (
    E5_EMBEDDING_DIM,
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
from .ingest_support import (
    _apply_test_mode_subset,
    _build_parent_ingest_records,
    _build_qdrant_point_id,
    _chunk_records,
    _ensure_qdrant_collection,
    _fetch_doc_statuses,
    _get_status_attr,
    _load_ingest_manifest,
    _mark_manifest_record,
    _normalize_doc_status_value,
    _resume_existing_pipeline_if_needed,
    _save_ingest_manifest,
    _validate_paths,
)
from .providers import _build_gemini_llm_func
from .runtime import EmbeddingFunc, LightRAG


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
    resolved_gemini_key = _require_gemini_key(gemini_key)
    resolved_gemini_model_name = _resolve_gemini_model_name(gemini_model_name)
    resolved_gemini_rpm_limit = _resolve_gemini_rpm_limit(
        resolved_gemini_model_name,
        gemini_requests_per_minute,
    )
    resolved_lightrag_batch_size = _resolve_lightrag_batch_size(
        lightrag_batch_size
    )
    resolved_lightrag_max_parallel_insert = (
        _resolve_lightrag_max_parallel_insert(
            lightrag_max_parallel_insert
        )
    )
    resolved_resume_existing_queue = _resolve_resume_existing_queue(
        resume_existing_queue
    )
    resolved_gemini_max_concurrency = _resolve_gemini_max_concurrency(
        gemini_max_concurrency or resolved_lightrag_max_parallel_insert
    )
    resolved_gemini_transient_max_retries = (
        _resolve_gemini_transient_max_retries(
            gemini_transient_max_retries
        )
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

    # Chuẩn bị dữ liệu gốc rồi dựng parent-child chunks như cũ.
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

    print(f"Đang tải {DENSE_MODEL_NAME}...")
    dense_model = E5EmbeddingModel(
        E5EmbeddingConfig(
            prompt_name=E5_PASSAGE_PROMPT_NAME,
            batch_size=32,
        )
    )
    sparse_model = SparseTextEmbedding(SPARSE_MODEL_NAME)

    # NHÁNH 1: Nạp child chunks vào Qdrant bằng id ổn định để tránh duplicate.
    print(f"\n[1/2] ĐANG NẠP {len(child_chunks)} CHILD CHUNKS VÀO QDRANT...")
    qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
    collection_state = _ensure_qdrant_collection(
        qdrant_client,
        collection_name,
        recreate_collection=recreate_collection,
    )
    print(
        "Qdrant collection state:",
        collection_state,
        f"(collection={collection_name})",
    )

    qdrant_batch_size = 256
    total_child_chunks = len(child_chunks)
    for index in range(0, total_child_chunks, qdrant_batch_size):
        batch_docs = child_chunks[index:index + qdrant_batch_size]
        batch_texts = [doc.page_content for doc in batch_docs]

        # Dùng dense + sparse embedding để giữ tương thích với query hybrid.
        batch_dense = dense_model.embed(batch_texts)
        batch_sparse = list(sparse_model.embed(batch_texts, parallel=0))

        points = []
        for batch_index, batch_doc in enumerate(batch_docs):
            points.append(
                models.PointStruct(
                    id=_build_qdrant_point_id(batch_doc),
                    vector={
                        "dense": batch_dense[batch_index].tolist(),
                        "sparse": models.SparseVector(
                            indices=batch_sparse[batch_index].indices.tolist(),
                            values=batch_sparse[batch_index].values.tolist(),
                        ),
                    },
                    payload={
                        "page_content": batch_doc.page_content,
                        **batch_doc.metadata,
                    },
                )
            )

        qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
        print(
            "Qdrant progress:",
            f"{min(index + len(batch_docs), total_child_chunks)}/{total_child_chunks}",
        )

    print(f"\n[2/2] ĐANG NẠP {len(parent_docs)} PARENT CHUNKS VÀO LIGHTRAG...")
    parent_records, duplicate_in_current_run = _build_parent_ingest_records(
        parent_docs
    )
    print(
        "LightRAG preflight:",
        f"records={len(parent_records)}, "
        f"deduped_within_run={duplicate_in_current_run}"
    )
    manifest = _load_ingest_manifest()

    async def custom_fastembed(texts: list[str]) -> np.ndarray:
        """Adapter embedding async để LightRAG tái sử dụng cùng dense model với Qdrant."""
        return dense_model.embed(texts)

    rag = LightRAG(
        working_dir=str(LIGHTRAG_WORKSPACE),
        llm_model_func=gemini_llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=E5_EMBEDDING_DIM,
            max_token_size=E5_MAX_LENGTH,
            func=custom_fastembed,
        ),
        chunk_token_size=5000,
        chunk_overlap_token_size=0,
        max_parallel_insert=resolved_lightrag_max_parallel_insert,
    )
    await rag.initialize_storages()

    try:
        await _resume_existing_pipeline_if_needed(
            rag,
            resume_existing_queue=resolved_resume_existing_queue,
        )

        existing_statuses = await _fetch_doc_statuses(
            rag,
            [record["doc_id"] for record in parent_records],
        )

        records_to_ingest = []
        skipped_existing = 0
        skipped_by_status: dict[str, int] = {}

        for record in parent_records:
            status_obj = existing_statuses.get(record["doc_id"])
            normalized_status = _normalize_doc_status_value(
                _get_status_attr(status_obj, "status")
            )

            if normalized_status:
                skipped_existing += 1
                skipped_by_status[normalized_status] = (
                    skipped_by_status.get(normalized_status, 0) + 1
                )
                note = _get_status_attr(status_obj, "error_msg", "") or (
                    "Bỏ qua vì doc_id đã tồn tại trong LightRAG"
                )
                _mark_manifest_record(
                    manifest,
                    record,
                    status=normalized_status,
                    track_id=_get_status_attr(status_obj, "track_id"),
                    note=note,
                )
                continue

            records_to_ingest.append(record)

        _save_ingest_manifest(manifest)

        print(
            "LightRAG dedup summary:",
            f"skip_existing={skipped_existing}, "
            f"new_records={len(records_to_ingest)}, "
            f"status_breakdown={skipped_by_status}"
        )

        if not records_to_ingest:
            print("Không còn parent chunk mới nào cần nạp vào LightRAG.")
            return

        batches = _chunk_records(records_to_ingest, resolved_lightrag_batch_size)
        total_batches = len(batches)

        for batch_index, batch_records in enumerate(batches, start=1):
            batch_track_id = (
                "insert_batch_"
                f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_"
                f"{batch_index:04d}"
            )
            batch_texts = [record["content"] for record in batch_records]
            batch_ids = [record["doc_id"] for record in batch_records]
            batch_file_paths = [record["file_path"] for record in batch_records]

            print(
                f"LightRAG batch {batch_index}/{total_batches}: "
                f"enqueue {len(batch_records)} docs "
                f"(track_id={batch_track_id})"
            )
            try:
                await rag.apipeline_enqueue_documents(
                    batch_texts,
                    ids=batch_ids,
                    file_paths=batch_file_paths,
                    track_id=batch_track_id,
                )
                await rag.apipeline_process_enqueue_documents()
            except Exception as exc:
                for record in batch_records:
                    _mark_manifest_record(
                        manifest,
                        record,
                        status="batch_failed",
                        track_id=batch_track_id,
                        note=str(exc),
                    )
                manifest["last_track_id"] = batch_track_id
                manifest["last_batch_failed_at"] = (
                    datetime.now(timezone.utc).isoformat()
                )
                _save_ingest_manifest(manifest)
                raise

            batch_statuses = await _fetch_doc_statuses(rag, batch_ids)
            batch_summary: dict[str, int] = {}
            for record in batch_records:
                status_obj = batch_statuses.get(record["doc_id"])
                normalized_status = _normalize_doc_status_value(
                    _get_status_attr(status_obj, "status")
                ) or "unknown"
                batch_summary[normalized_status] = (
                    batch_summary.get(normalized_status, 0) + 1
                )
                _mark_manifest_record(
                    manifest,
                    record,
                    status=normalized_status,
                    track_id=batch_track_id,
                    note=_get_status_attr(status_obj, "error_msg", "") or "",
                )

            manifest["last_track_id"] = batch_track_id
            manifest["last_batch_completed_at"] = (
                datetime.now(timezone.utc).isoformat()
            )
            _save_ingest_manifest(manifest)
            print(
                f"LightRAG batch {batch_index}/{total_batches} hoàn tất: "
                f"{batch_summary}"
            )

        print("HOÀN TẤT NẠP KÉP THÀNH CÔNG!")
    finally:
        await rag.finalize_storages()


__all__ = ["hybrid_ingest"]
