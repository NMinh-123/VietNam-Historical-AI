"""Truy xuất hybrid Qdrant: dense (E5) + sparse (BM25) → RRF → BGE rerank."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from app.core.embeddings.embedder import E5EmbeddingModel
from app.core.utils.helpers import build_query, build_source_label
from app.core.vectordb.vector_store import (
    COLLECTION_NAME,
    PARENT_COLLECTION_NAME,
    fetch_parent_texts,
)

_logger = logging.getLogger(__name__)

from app.core.app_config import get_config as _get_config

_ret_cfg = _get_config().retrieval
_MAX_CHUNKS_PER_PARENT = _ret_cfg.max_chunks_per_parent

# ── BGE Reranker (lazy load lần đầu dùng) ────────────────────────────────────

_reranker = None
# MiniLM-L12: ~67MB, ~0.8s/query trên CPU — nhanh hơn bge-reranker-v2-m3 (~500×)
# Hạn chế: trained trên MS MARCO (English). Với câu hỏi tiếng Việt phức tạp có thể dùng
# "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1" (multilingual) nếu cần chính xác hơn.
_RERANKER_MODEL = "BAAI/bge-reranker-base"


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _logger.info("Đang tải cross-encoder reranker: %s", _RERANKER_MODEL)
        _reranker = CrossEncoder(_RERANKER_MODEL)
        _logger.info("Reranker sẵn sàng.")
    return _reranker


def _retrieve(
    query: str,
    top_k: int,
    limit: int,
    qdrant: QdrantClient,
    dense_model: E5EmbeddingModel,
    sparse_model: SparseTextEmbedding,
    parent_collection: str = PARENT_COLLECTION_NAME,
) -> dict[str, Any]:
    """Hybrid search (RRF) → dedup by parent → BGE rerank → top_k."""
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

    # BGE chấm trên child chunks (~600 tokens) — nhanh hơn 3× so với parent (2000 tokens)
    # Dedup theo parent_id: giữ child chunk có RRF score cao nhất mỗi parent
    seen_parents: dict[str, dict[str, Any]] = {}
    for hit in result.points:
        payload = hit.payload or {}
        parent_id = payload.get("parent_id")
        child_text = payload.get("page_content", "")
        if not child_text or not parent_id:
            continue
        key = parent_id
        rrf_score = float(hit.score or 0.0)
        if key not in seen_parents or rrf_score > seen_parents[key]["rrf_score"]:
            seen_parents[key] = {
                "child_text": child_text,
                "rrf_score": rrf_score,
                "parent_id": parent_id,
                "source": payload.get("source"),
                "page": payload.get("page"),
                "page_label": payload.get("page_label"),
                "title": payload.get("title"),
            }

    candidates = list(seen_parents.values())

    # BGE cross-encoder rerank trên child chunks
    reranker = _get_reranker()
    pairs = [(query, item["child_text"]) for item in candidates]
    bge_scores = reranker.predict(pairs)
    if not hasattr(bge_scores, "__iter__"):
        bge_scores = [bge_scores]

    import gc
    gc.collect()
        
    for item, bge_score in zip(candidates, bge_scores):
        item["score"] = float(bge_score)

    ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)

    # Fetch parent texts chỉ cho top_k candidates đã được chọn
    top_parent_ids = []
    seen: dict[str, int] = {}
    for item in ranked:
        pid = item["parent_id"]
        if seen.get(pid, 0) < _MAX_CHUNKS_PER_PARENT:
            top_parent_ids.append(pid)
            seen[pid] = seen.get(pid, 0) + 1
        if len(top_parent_ids) >= top_k:
            break

    parent_texts = fetch_parent_texts(qdrant, top_parent_ids, parent_collection)

    final: list[dict[str, Any]] = []
    seen2: dict[str, int] = {}
    for item in ranked:
        pid = item["parent_id"]
        if seen2.get(pid, 0) >= _MAX_CHUNKS_PER_PARENT:
            continue
        context = parent_texts.get(pid) or item["child_text"]
        item["text"] = context
        item["source_label"] = build_source_label(item)
        final.append(item)
        seen2[pid] = seen2.get(pid, 0) + 1
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
