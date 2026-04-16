"""Nhánh index Qdrant: embed child chunks và upsert vào vector database."""

from __future__ import annotations

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from data.process_data.e5_embeddings import E5EmbeddingModel
from services.chatbot.index.ingest_support import _build_qdrant_point_id, _ensure_qdrant_collection


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

    Trước khi embed, kiểm tra ID nào đã tồn tại trong collection để bỏ qua,
    tránh tính toán lại vector cho dữ liệu trùng lặp.
    """
    # Tạo hoặc tái sử dụng collection theo cấu hình
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

    total = len(child_chunks)
    total_skipped = 0
    total_upserted = 0

    for batch_start in range(0, total, batch_size):
        batch_docs = child_chunks[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1

        # Tính ID ổn định cho từng doc trong batch
        batch_ids = [_build_qdrant_point_id(doc) for doc in batch_docs]

        # Xác định ID nào đã tồn tại để loại bỏ dữ liệu trùng lặp
        already_exists = qdrant_client.retrieve(
            collection_name=collection_name,
            ids=batch_ids,
            with_payload=False,
            with_vectors=False,
        )
        existing_ids = {str(point.id) for point in already_exists}

        # Chỉ giữ lại những doc có ID chưa tồn tại
        new_pairs = [
            (doc, pid)
            for doc, pid in zip(batch_docs, batch_ids)
            if pid not in existing_ids
        ]

        n_skipped = len(batch_docs) - len(new_pairs)
        total_skipped += n_skipped

        if not new_pairs:
            print(
                f"Qdrant batch {batch_num}: "
                f"bỏ qua toàn bộ {n_skipped} chunk (đã tồn tại)."
            )
            continue

        # Chỉ embed những doc thực sự mới, tiết kiệm thời gian inference
        new_docs, new_ids = zip(*new_pairs)
        batch_texts = [doc.page_content for doc in new_docs]

        batch_dense = dense_model.embed(batch_texts)
        batch_sparse = list(sparse_model.embed(batch_texts, parallel=0))

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
                payload={
                    "page_content": doc.page_content,
                    **doc.metadata,
                },
            )
            for i, (doc, pid) in enumerate(zip(new_docs, new_ids))
        ]

        qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

        total_upserted += len(points)
        done = min(batch_start + len(batch_docs), total)
        print(
            f"Qdrant batch {batch_num}: "
            f"upsert {len(points)} mới, bỏ qua {n_skipped} trùng | "
            f"tiến độ {done}/{total}"
        )

    print(
        f"\nQdrant hoàn tất: {total_upserted} upserted, "
        f"{total_skipped} bỏ qua (trùng lặp)."
    )


__all__ = ["ingest_to_qdrant"]
