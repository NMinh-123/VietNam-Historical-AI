"""Helper cho ingest: validate path, dedup, manifest và resume queue."""

from __future__ import annotations

import json
from hashlib import md5
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from qdrant_client import QdrantClient, models

from data.process_data.e5_embeddings import E5_EMBEDDING_DIM
from .config import (
    LIGHTRAG_INGEST_MANIFEST_PATH,
    LIGHTRAG_WORKSPACE,
    PARENT_DOCSTORE_PATH,
    QDRANT_DB_PATH,
    RAW_DATA_PATH,
)
from .runtime import DocStatus, compute_mdhash_id, sanitize_text_for_encoding


def _validate_paths() -> None:
    """Chuẩn bị sẵn các thư mục cần ghi trước khi chạy ingest."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục raw_data tại: {RAW_DATA_PATH}"
        )

    PARENT_DOCSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIGHTRAG_WORKSPACE.mkdir(parents=True, exist_ok=True)
    QDRANT_DB_PATH.mkdir(parents=True, exist_ok=True)


def _apply_test_mode_subset(
    parent_docs: list,
    child_chunks: list,
    parent_store: dict,
    parent_limit: int,
) -> tuple[list, list, dict]:
    """Cắt nhỏ dataset để test nhanh nhưng vẫn giữ đúng parent-child mapping."""
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
) -> None:
    """Tạo mới collection với schema dense+sparse đúng cho hybrid search."""
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


def _ensure_qdrant_collection(
    client: QdrantClient,
    collection_name: str,
    recreate_collection: bool,
) -> str:
    """Tạo collection nếu chưa có, tái tạo nếu recreate=True, hoặc tái sử dụng."""
    exists = client.collection_exists(collection_name)

    if exists and recreate_collection:
        print(f"Xóa collection Qdrant cũ: {collection_name}")
        client.delete_collection(collection_name)
        exists = False

    if not exists:
        _prepare_qdrant_collection(client, collection_name)
        return "recreated" if recreate_collection else "created"

    return "reused"


def _load_ingest_manifest() -> dict:
    """Đọc manifest ingest từ đĩa; nếu lỗi hoặc thiếu thì trả về cấu trúc rỗng an toàn."""
    if not LIGHTRAG_INGEST_MANIFEST_PATH.exists():
        return {"records": {}}

    try:
        with open(LIGHTRAG_INGEST_MANIFEST_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as exc:
        print(f"Cảnh báo: Không đọc được ingest manifest: {exc}")
        return {"records": {}}

    if not isinstance(data, dict):
        return {"records": {}}

    records = data.get("records")
    if not isinstance(records, dict):
        data["records"] = {}
    return data


def _save_ingest_manifest(manifest: dict) -> None:
    """Ghi manifest ingest ra đĩa để checkpoint trạng thái sau mỗi batch."""
    LIGHTRAG_INGEST_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LIGHTRAG_INGEST_MANIFEST_PATH, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def _get_status_attr(status_obj, key: str, default=None):
    """Đọc thuộc tính trạng thái theo cả hai kiểu object hoặc dict."""
    if isinstance(status_obj, dict):
        return status_obj.get(key, default)
    return getattr(status_obj, key, default)


def _normalize_doc_status_value(raw_status) -> str:
    """Chuẩn hóa giá trị status về chuỗi chữ thường."""
    if raw_status is None:
        return ""
    if hasattr(raw_status, "value"):
        return str(raw_status.value).lower()
    return str(raw_status).lower()


def _chunk_records(items: list[dict], batch_size: int) -> list[list[dict]]:
    """Chia danh sách record thành nhiều lô nhỏ để ingest tuần tự theo checkpoint."""
    return [
        items[index:index + batch_size]
        for index in range(0, len(items), batch_size)
    ]


def _normalize_file_path(raw_path: object) -> str:
    """Chuẩn hóa đường dẫn nguồn về chuỗi hợp lệ; thiếu thì gắn nhãn mặc định."""
    if isinstance(raw_path, str) and raw_path.strip():
        return raw_path.strip()
    return "unknown_source"


def _build_qdrant_point_id(document) -> str:
    """Tạo UUID ổn định để Qdrant remote chấp nhận và ingest lặp lại không nhân bản."""
    metadata = dict(getattr(document, "metadata", {}) or {})
    stable_child_id = metadata.get("child_id") or metadata.get("doc_id")
    if isinstance(stable_child_id, str) and stable_child_id.strip():
        stable_hex = md5(stable_child_id.strip().encode("utf-8")).hexdigest()
        return str(UUID(stable_hex))

    signature = json.dumps(
        {
            "page_content": getattr(document, "page_content", "") or "",
            "parent_id": metadata.get("parent_id"),
            "source": metadata.get("source"),
            "page": metadata.get("page"),
            "page_label": metadata.get("page_label"),
            "title": metadata.get("title"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    stable_hex = md5(signature.encode("utf-8")).hexdigest()
    return str(UUID(stable_hex))


def _build_parent_ingest_records(parent_docs: list) -> tuple[list[dict], int]:
    """Sinh record ingest có doc_id ổn định để tránh duplicate theo content."""
    records_by_doc_id: dict[str, dict] = {}
    duplicate_count = 0
    empty_count = 0

    for parent_doc in parent_docs:
        cleaned_content = sanitize_text_for_encoding(
            parent_doc.page_content or ""
        ).strip()
        if not cleaned_content:
            empty_count += 1
            continue

        metadata = dict(parent_doc.metadata or {})
        stable_doc_id = compute_mdhash_id(cleaned_content, prefix="doc-")
        file_path = _normalize_file_path(metadata.get("source"))
        original_parent_id = metadata.get("doc_id")

        if stable_doc_id in records_by_doc_id:
            duplicate_count += 1
            provenance = records_by_doc_id[stable_doc_id]["original_parent_ids"]
            if original_parent_id and original_parent_id not in provenance:
                provenance.append(original_parent_id)
            continue

        records_by_doc_id[stable_doc_id] = {
            "doc_id": stable_doc_id,
            "content": cleaned_content,
            "file_path": file_path,
            "title": metadata.get("title"),
            "page": metadata.get("page"),
            "page_label": metadata.get("page_label"),
            "source": file_path,
            "original_parent_ids": [original_parent_id] if original_parent_id else [],
        }

    if empty_count:
        print(f"[WARNING] Bỏ qua {empty_count} parent doc có nội dung rỗng sau khi làm sạch.")

    return list(records_by_doc_id.values()), duplicate_count


async def _fetch_doc_statuses(
    rag,
    doc_ids: list[str],
    lookup_batch_size: int = 200,
) -> dict[str, object]:
    """Đọc trạng thái theo lô để tránh query quá to vào doc_status storage."""
    found_statuses: dict[str, object] = {}
    for index in range(0, len(doc_ids), lookup_batch_size):
        batch_ids = doc_ids[index:index + lookup_batch_size]
        if not batch_ids:
            continue
        batch_statuses = await rag.aget_docs_by_ids(batch_ids)
        found_statuses.update(batch_statuses)
    return found_statuses


async def _resume_existing_pipeline_if_needed(
    rag,
    *,
    resume_existing_queue: bool,
) -> None:
    """Drain queue cũ trước để tránh trộn batch mới vào trạng thái bẩn."""
    pending_docs = await rag.get_docs_by_status(DocStatus.PENDING)
    processing_docs = await rag.get_docs_by_status(DocStatus.PROCESSING)

    pending_count = len(pending_docs)
    processing_count = len(processing_docs)
    if pending_count == 0 and processing_count == 0:
        return

    message = (
        "Phát hiện queue LightRAG chưa xử lý xong: "
        f"pending={pending_count}, processing={processing_count}."
    )
    if not resume_existing_queue:
        raise RuntimeError(
            message
            + " Dừng để tránh trộn batch mới với queue cũ. "
            "Đặt LIGHTRAG_RESUME_EXISTING_QUEUE=1 nếu muốn resume queue hiện tại."
        )

    print(message + " Đang resume queue cũ trước khi nạp batch mới...")
    await rag.apipeline_process_enqueue_documents()

    pending_docs = await rag.get_docs_by_status(DocStatus.PENDING)
    processing_docs = await rag.get_docs_by_status(DocStatus.PROCESSING)
    if pending_docs or processing_docs:
        raise RuntimeError(
            "Queue LightRAG cũ vẫn chưa drain hết sau khi resume "
            f"(pending={len(pending_docs)}, processing={len(processing_docs)}). "
            "Dừng để tránh trộn batch mới với trạng thái chưa ổn định."
        )


def _mark_manifest_record(
    manifest: dict,
    record: dict,
    *,
    status: str,
    track_id: str | None = None,
    note: str | None = None,
) -> None:
    """Lưu lại trạng thái ingest để lần sau có thể audit hoặc resume an toàn."""
    records = manifest.setdefault("records", {})
    records[record["doc_id"]] = {
        "doc_id": record["doc_id"],
        "file_path": record["file_path"],
        "title": record.get("title"),
        "page": record.get("page"),
        "page_label": record.get("page_label"),
        "source": record.get("source"),
        "original_parent_ids": record.get("original_parent_ids", []),
        "status": status,
        "track_id": track_id,
        "note": note or "",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


__all__ = [
    "_apply_test_mode_subset",
    "_build_qdrant_point_id",
    "_build_parent_ingest_records",
    "_chunk_records",
    "_ensure_qdrant_collection",
    "_fetch_doc_statuses",
    "_get_status_attr",
    "_load_ingest_manifest",
    "_mark_manifest_record",
    "_normalize_doc_status_value",
    "_prepare_qdrant_collection",
    "_resume_existing_pipeline_if_needed",
    "_save_ingest_manifest",
    "_validate_paths",
    "_normalize_file_path",
]
