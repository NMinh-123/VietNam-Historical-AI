"""Tìm kiếm hybrid Qdrant và tái xếp hạng lexical cho pipeline truy hồi."""

from __future__ import annotations

import asyncio
from typing import Any

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from data.process_data.e5_embeddings import E5EmbeddingModel
from services.chatbot.index import COLLECTION_NAME
from .context_builder import _build_source_label
from .text_utils import _lexical_score, build_query

# Trọng số hybrid scoring: fused (RRF) và lexical
_FUSED_WEIGHT = 0.7
_LEXICAL_WEIGHT = 0.3

# Hệ số thưởng cho parent_id xuất hiện nhiều lần trong kết quả
_COUNT_BONUS = 0.2


def _retrieve(
    query: str,
    top_k: int,
    limit: int,
    qdrant: QdrantClient,
    dense_model: E5EmbeddingModel,
    sparse_model: SparseTextEmbedding,
    parent_store: dict[str, str],
) -> dict[str, Any]:
    # Tìm kiếm hybrid (dense + sparse) trên Qdrant, tái xếp hạng bằng điểm lexical, trả top_k kết quả đa dạng
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

    grouped: dict[str, dict[str, Any]] = {}

    for hit in result.points:
        payload = hit.payload or {}
        parent_id = payload.get("parent_id")

        context = parent_store.get(parent_id) or payload.get("page_content", "")
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

    # Đa dạng hoá: tối đa 2 chunk cho mỗi parent_id
    final: list[dict[str, Any]] = []
    seen: dict[str, int] = {}

    for item in ranked:
        pid = item["parent_id"]
        if seen.get(pid, 0) < 2:
            item["source_label"] = _build_source_label(item)
            final.append(item)
            seen[pid] = seen.get(pid, 0) + 1
        if len(final) >= top_k:
            break

    return {"items": final}


async def get_vector(
    query: str,
    top_k: int,
    limit: int,
    qdrant: QdrantClient,
    dense_model: E5EmbeddingModel,
    sparse_model: SparseTextEmbedding,
    parent_store: dict[str, str],
) -> dict[str, Any]:
    # Chạy _retrieve trên thread riêng để không chặn event loop
    return await asyncio.to_thread(
        _retrieve, query, top_k, limit, qdrant, dense_model, sparse_model, parent_store
    )


__all__ = ["_retrieve", "get_vector"]
