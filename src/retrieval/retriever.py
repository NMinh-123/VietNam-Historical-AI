"""Truy xuất hybrid Qdrant: dense (E5) + sparse (BM25) → RRF → lexical rerank."""

from __future__ import annotations

import asyncio
from typing import Any

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from src.embeddings.embedder import E5EmbeddingModel
from src.utils.helpers import _lexical_score, build_query, build_source_label
from src.vectordb.vector_store import (
    COLLECTION_NAME,
    PARENT_COLLECTION_NAME,
    fetch_parent_texts,
)

# ── Trọng số scoring ──────────────────────────────────────────────────────────

from src.app_config import get_config as _get_config

_ret_cfg = _get_config().retrieval
_FUSED_WEIGHT = _ret_cfg.fused_weight
_LEXICAL_WEIGHT = _ret_cfg.lexical_weight
_COUNT_BONUS = _ret_cfg.count_bonus
_MAX_CHUNKS_PER_PARENT = _ret_cfg.max_chunks_per_parent


def _retrieve(
    query: str,
    top_k: int,
    limit: int,
    qdrant: QdrantClient,
    dense_model: E5EmbeddingModel,
    sparse_model: SparseTextEmbedding,
    parent_collection: str = PARENT_COLLECTION_NAME,
) -> dict[str, Any]:
    """Hybrid search + lexical rerank, trả về top_k kết quả đa dạng."""
    q = build_query(query)

    dense_vec = dense_model.embed([q["dense"]])[0]
    sparse_vec = list(sparse_model.embed([q["sparse"]]))[0]

    result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense_vec.tolist(), using="dense", limit=limit),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
                using="sparse",
                limit=limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
    )

    unique_parent_ids = list({
        (hit.payload or {}).get("parent_id")
        for hit in result.points
        if (hit.payload or {}).get("parent_id")
    })
    parent_texts = fetch_parent_texts(qdrant, unique_parent_ids, parent_collection)

    grouped: dict[str, dict[str, Any]] = {}
    for hit in result.points:
        payload = hit.payload or {}
        parent_id = payload.get("parent_id")
        context = parent_texts.get(parent_id) or payload.get("page_content", "")
        if not context:
            continue

        fused = float(hit.score or 0.0)
        lexical = _lexical_score(q["keywords"], context)
        score = _FUSED_WEIGHT * fused + _LEXICAL_WEIGHT * lexical
        key = parent_id or context[:100]

        if key not in grouped:
            grouped[key] = {
                "text": context,
                "score": score,
                "parent_id": parent_id,
                "count": 1,
                "source": payload.get("source"),
                "page": payload.get("page"),
                "page_label": payload.get("page_label"),
                "title": payload.get("title"),
            }
        else:
            grouped[key]["score"] += score
            grouped[key]["count"] += 1

    ranked = sorted(
        grouped.values(),
        key=lambda x: x["score"] + _COUNT_BONUS * x["count"],
        reverse=True,
    )

    final: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for item in ranked:
        pid = item["parent_id"]
        if seen.get(pid, 0) < _MAX_CHUNKS_PER_PARENT:
            item["source_label"] = build_source_label(item)
            final.append(item)
            seen[pid] = seen.get(pid, 0) + 1
        if len(final) >= top_k:
            break

    return {"items": final}


async def retrieve(
    query: str,
    top_k: int,
    limit: int,
    qdrant: QdrantClient,
    dense_model: E5EmbeddingModel,
    sparse_model: SparseTextEmbedding,
    parent_collection: str = PARENT_COLLECTION_NAME,
) -> dict[str, Any]:
    """Async wrapper — chạy _retrieve trên thread riêng để không chặn event loop."""
    return await asyncio.to_thread(
        _retrieve, query, top_k, limit, qdrant, dense_model, sparse_model, parent_collection
    )


__all__ = ["_retrieve", "retrieve"]
