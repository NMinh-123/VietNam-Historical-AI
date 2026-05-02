"""Nhánh index LightRAG: ingest parent chunks vào knowledge graph."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import numpy as np

from data.process_data.e5_embeddings import (
    E5_EMBEDDING_DIM,
    E5_MAX_LENGTH,
    E5EmbeddingModel,
)
from .ingest_support import (
    _build_parent_ingest_records,
    _chunk_records,
    _fetch_doc_statuses,
    _get_status_attr,
    _load_ingest_manifest,
    _mark_manifest_record,
    _normalize_doc_status_value,
    _resume_existing_pipeline_if_needed,
    _save_ingest_manifest,
)
from .runtime import EmbeddingFunc, LightRAG


def build_lightrag_instance(
    dense_model: E5EmbeddingModel,
    llm_func,
    working_dir: str,
    max_parallel_insert: int,
) -> LightRAG:
    """Khởi tạo LightRAG với embedding adapter chạy trên thread riêng để không block event loop."""
    async def _embed_func(texts: list[str]) -> np.ndarray:
        # Dùng asyncio.to_thread vì dense_model.embed là hàm đồng bộ chạy trên CPU
        return await asyncio.to_thread(dense_model.embed, texts)

    return LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=E5_EMBEDDING_DIM,
            max_token_size=E5_MAX_LENGTH,
            func=_embed_func,
        ),
        chunk_token_size=5000,
        chunk_overlap_token_size=0,
        max_parallel_insert=max_parallel_insert,
    )


async def ingest_to_lightrag(
    parent_docs: list,
    rag: LightRAG,
    batch_size: int,
    resume_existing_queue: bool,
) -> None:
    """Dedup, enqueue và xử lý parent docs vào LightRAG knowledge graph.

    Dedup hoạt động ở hai tầng:
    - Tầng 1: loại doc trùng nội dung trong cùng một lần chạy (md5 hash).
    - Tầng 2: loại doc đã có trong LightRAG storage theo doc_id (bỏ qua status đã xử lý).
    """
    parent_records, duplicate_in_run = _build_parent_ingest_records(parent_docs)
    print(
        "LightRAG preflight:",
        f"records={len(parent_records)}, "
        f"trùng_trong_lần_chạy={duplicate_in_run}",
    )

    await rag.initialize_storages()
    manifest = _load_ingest_manifest()

    try:
        await _resume_existing_pipeline_if_needed(
            rag,
            resume_existing_queue=resume_existing_queue,
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
            "LightRAG dedup tổng kết: "
            f"trùng_storage={skipped_existing}, "
            f"mới_cần_nạp={len(records_to_ingest)}, "
            f"phân_loại={skipped_by_status}"
        )

        if not records_to_ingest:
            print("Không còn parent chunk mới nào cần nạp vào LightRAG.")
            return

        batches = _chunk_records(records_to_ingest, batch_size)
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

    finally:
        await rag.finalize_storages()


__all__ = ["build_lightrag_instance", "ingest_to_lightrag"]
