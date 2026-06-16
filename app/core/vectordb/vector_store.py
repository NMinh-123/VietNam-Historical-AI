"""Lưu & tìm kiếm vector — Qdrant hybrid (dense + sparse) với parent-child store."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from hashlib import md5
from typing import Any
from uuid import UUID

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from app.core.embeddings.embedder import E5EmbeddingModel, E5_EMBEDDING_DIM

# ── Collection constants ──────────────────────────────────────────────────────

from app.core.app_config import get_config as _get_config

_vdb_cfg = _get_config().vectordb
COLLECTION_NAME = _vdb_cfg.collection_name
PARENT_COLLECTION_NAME = _vdb_cfg.parent_collection_name


# ── ID helpers ────────────────────────────────────────────────────────────────

def parent_id_to_uuid(parent_id: str) -> str:
    """Chuyển parent_id dạng chuỗi thành UUID ổn định để dùng làm Qdrant point ID."""
    return str(UUID(md5(parent_id.encode("utf-8")).hexdigest()))


def build_qdrant_point_id(document: Any) -> str:
    """Tạo UUID ổn định từ child_id hoặc nội dung document."""
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
    return str(UUID(md5(signature.encode("utf-8")).hexdigest()))


# ── Collection management ─────────────────────────────────────────────────────

def _prepare_collection(client: QdrantClient, collection_name: str) -> None:
    """Tạo collection với schema dense+sparse cho hybrid search."""
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


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    recreate: bool = False,
) -> str:
    """Tạo collection nếu chưa có, tái tạo nếu recreate=True, hoặc tái sử dụng."""
    exists = client.collection_exists(collection_name)
    if exists and recreate:
        print(f"Xóa collection Qdrant cũ: {collection_name}")
        client.delete_collection(collection_name)
        exists = False
    if not exists:
        _prepare_collection(client, collection_name)
        return "recreated" if recreate else "created"
    return "reused"


def ensure_parent_collection(
    client: QdrantClient,
    collection_name: str = PARENT_COLLECTION_NAME,
    recreate: bool = False,
) -> str:
    """Tạo collection payload-only cho parent docs (không cần vector)."""
    exists = client.collection_exists(collection_name)
    if exists and recreate:
        client.delete_collection(collection_name)
        exists = False
    if not exists:
        client.create_collection(collection_name=collection_name, vectors_config={})
        return "recreated" if recreate else "created"
    return "reused"


# ── Ingest: child chunks → Qdrant (dense + sparse) ───────────────────────────

async def ingest_to_qdrant(
    child_chunks: list,
    qdrant_client: QdrantClient,
    collection_name: str,
    dense_model: E5EmbeddingModel,
    sparse_model: SparseTextEmbedding,
    batch_size: int = 256,
    recreate_collection: bool = False,
) -> None:
    """Nhúng child chunks bằng E5 + BM25 rồi upsert vào Qdrant.

    Kiểm tra ID đã tồn tại trước khi embed để bỏ qua chunk trùng lặp.
    """
    state = ensure_collection(qdrant_client, collection_name, recreate=recreate_collection)
    print(f"Qdrant collection state: {state} (collection={collection_name})")

    total = len(child_chunks)
    total_skipped = 0
    total_upserted = 0
    start_time = time.monotonic()

    for batch_start in range(0, total, batch_size):
        batch_docs = child_chunks[batch_start: batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        batch_ids = [build_qdrant_point_id(doc) for doc in batch_docs]

        already_exists = await asyncio.to_thread(
            qdrant_client.retrieve,
            collection_name=collection_name,
            ids=batch_ids,
            with_payload=False,
            with_vectors=False,
        )
        existing_ids = {str(point.id) for point in already_exists}
        new_pairs = [(doc, pid) for doc, pid in zip(batch_docs, batch_ids) if pid not in existing_ids]

        n_skipped = len(batch_docs) - len(new_pairs)
        total_skipped += n_skipped
        done = min(batch_start + len(batch_docs), total)
        pct = done / total * 100
        elapsed = time.monotonic() - start_time

        if not new_pairs:
            print(f"[{pct:5.1f}%] batch {batch_num}: bỏ qua {n_skipped} chunk (đã tồn tại) | {done}/{total} | {elapsed:.0f}s")
            continue

        new_docs, new_ids = zip(*new_pairs)
        batch_texts = [doc.page_content for doc in new_docs]

        batch_dense = await asyncio.to_thread(dense_model.embed, batch_texts)
        batch_sparse = await asyncio.to_thread(lambda: list(sparse_model.embed(batch_texts, parallel=0)))

        points = [
            models.PointStruct(
                id=pid,
                vector={
                    "dense": batch_dense[i].tolist(),
                    "sparse": models.SparseVector(
                        indices=batch_sparse[i].indices.tolist(),
                        values=batch_sparse[i].values.tolist(),
                    ),
                },
                payload={"page_content": doc.page_content, **doc.metadata},
            )
            for i, (doc, pid) in enumerate(zip(new_docs, new_ids))
        ]

        await asyncio.to_thread(qdrant_client.upsert, collection_name=collection_name, points=points, wait=True)
        total_upserted += len(points)
        elapsed = time.monotonic() - start_time
        print(f"[{pct:5.1f}%] batch {batch_num}: upsert {len(points)} mới, skip {n_skipped} trùng | {done}/{total} | {elapsed:.0f}s")

    total_elapsed = time.monotonic() - start_time
    print(f"\nQdrant hoàn tất: {total_upserted} upserted, {total_skipped} bỏ qua | tổng {total_elapsed:.0f}s")


# ── Ingest: parent docs → Qdrant (payload-only) ───────────────────────────────

async def ingest_parents_to_qdrant(
    parent_store: dict[str, str],
    qdrant_client: QdrantClient,
    collection_name: str,
    recreate_collection: bool = False,
    batch_size: int = 512,
) -> None:
    """Upsert parent docs vào Qdrant collection payload-only (không cần vector)."""
    state = ensure_parent_collection(qdrant_client, collection_name, recreate=recreate_collection)
    print(f"Parent collection state: {state} (collection={collection_name})")

    items = list(parent_store.items())
    total = len(items)
    total_upserted = 0
    start_time = time.monotonic()

    for batch_start in range(0, total, batch_size):
        batch = items[batch_start: batch_start + batch_size]
        batch_ids = [parent_id_to_uuid(pid) for pid, _ in batch]

        already_exists = await asyncio.to_thread(
            qdrant_client.retrieve,
            collection_name=collection_name,
            ids=batch_ids,
            with_payload=False,
            with_vectors=False,
        )
        existing_ids = {str(p.id) for p in already_exists}
        new_pairs = [(pid, text, uid) for (pid, text), uid in zip(batch, batch_ids) if uid not in existing_ids]

        n_skipped = len(batch) - len(new_pairs)
        done = min(batch_start + len(batch), total)
        pct = done / total * 100

        if not new_pairs:
            print(f"[{pct:5.1f}%] bỏ qua {n_skipped} parent (đã tồn tại) | {done}/{total}")
            continue

        points = [
            models.PointStruct(id=uid, vector={}, payload={"parent_id": pid, "content": text})
            for pid, text, uid in new_pairs
        ]
        await asyncio.to_thread(qdrant_client.upsert, collection_name=collection_name, points=points, wait=True)
        total_upserted += len(points)
        elapsed = time.monotonic() - start_time
        print(f"[{pct:5.1f}%] upsert {len(points)} mới, skip {n_skipped} trùng | {done}/{total} | {elapsed:.0f}s")

    elapsed = time.monotonic() - start_time
    print(f"\nParent ingest hoàn tất: {total_upserted} upserted | tổng {elapsed:.0f}s")


# ── Fetch parent texts ────────────────────────────────────────────────────────

def fetch_parent_texts(
    qdrant: QdrantClient,
    parent_ids: list[str],
    parent_collection: str,
) -> dict[str, str]:
    """Batch-fetch nội dung parent docs từ Qdrant, trả về {parent_id: content}."""
    if not parent_ids:
        return {}
    uuids = [parent_id_to_uuid(pid) for pid in parent_ids]
    try:
        points = qdrant.retrieve(
            collection_name=parent_collection,
            ids=uuids,
            with_payload=True,
            with_vectors=False,
        )
        return {p.payload["parent_id"]: p.payload["content"] for p in points if p.payload}
    except Exception:
        return {}


__all__ = [
    "COLLECTION_NAME",
    "PARENT_COLLECTION_NAME",
    "build_qdrant_point_id",
    "ensure_collection",
    "ensure_parent_collection",
    "fetch_parent_texts",
    "ingest_parents_to_qdrant",
    "ingest_to_qdrant",
    "parent_id_to_uuid",
]
