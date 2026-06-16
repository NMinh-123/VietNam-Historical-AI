"""Cắt tài liệu thành parent/child chunks với semantic splitting tùy chọn."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CHUNK_ID_REGISTRY_PATH = CURRENT_DIR.parent.parent.parent / "data" / "chunk_id_registry.json"

from app.core.utils.helpers import clean_text, clean_documents  # noqa: E402


# ── Helper Functions ──────────────────────────────────────────────────────────


def _to_json_safe_metadata(metadata: dict) -> dict:
    safe_metadata = {}
    for key, value in (metadata or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe_metadata[key] = value
        elif isinstance(value, (list, tuple)):
            safe_metadata[key] = [
                item if isinstance(item, (str, int, float, bool)) or item is None else str(item)
                for item in value
            ]
        else:
            safe_metadata[key] = str(value)
    return safe_metadata


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _load_chunk_id_registry(registry_path: str | os.PathLike | None = None) -> dict:
    resolved_path = Path(registry_path or DEFAULT_CHUNK_ID_REGISTRY_PATH)
    default_registry = {
        "parent": {"next_index": 1, "signatures": {}},
        "child": {"next_index": 1, "signatures": {}},
    }
    if not resolved_path.exists():
        return default_registry
    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return default_registry
    if not isinstance(data, dict):
        return default_registry
    for chunk_type in ("parent", "child"):
        bucket = data.get(chunk_type)
        if not isinstance(bucket, dict):
            data[chunk_type] = {"next_index": 1, "signatures": {}}
            continue
        if not isinstance(bucket.get("next_index"), int) or bucket["next_index"] < 1:
            bucket["next_index"] = 1
        if not isinstance(bucket.get("signatures"), dict):
            bucket["signatures"] = {}
    return data


def _save_chunk_id_registry(
    registry: dict,
    registry_path: str | os.PathLike | None = None,
) -> None:
    resolved_path = Path(registry_path or DEFAULT_CHUNK_ID_REGISTRY_PATH)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def _build_chunk_signature_key(payload: dict) -> str:
    raw_signature = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return sha256(raw_signature.encode("utf-8")).hexdigest()


def _assign_stable_chunk_id(registry: dict, chunk_type: str, signature_key: str) -> str:
    bucket = registry.setdefault(chunk_type, {"next_index": 1, "signatures": {}})
    signatures = bucket.setdefault("signatures", {})
    existing_id = signatures.get(signature_key)
    if existing_id:
        return existing_id
    next_index = int(bucket.setdefault("next_index", 1))
    assigned_id = f"{chunk_type}_{next_index:06d}"
    signatures[signature_key] = assigned_id
    bucket["next_index"] = next_index + 1
    return assigned_id


# ── Character-based Splitting ─────────────────────────────────────────────────

def _split_long_text(text: str, max_chars: int) -> list[str]:
    text = _normalize_whitespace(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?;:])\s+(?=[A-ZÀ-Ỹ0-9])", text)
    if len(sentences) == 1:
        sentences = re.split(r"(?<=,)\s+", text)
    if len(sentences) == 1:
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    chunks, current_chunk, current_len = [], [], 0
    for sentence in sentences:
        sentence_len = len(sentence)
        if current_len + sentence_len > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_len = [], 0
        current_chunk.append(sentence)
        current_len += sentence_len + 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def _create_chunks_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    base_chunks = _split_long_text(text, max_chars=chunk_size // 2)
    final_chunks, current_text = [], ""
    for chunk in base_chunks:
        if not current_text:
            current_text = chunk
            continue
        combined = current_text + " " + chunk
        if len(combined) > chunk_size:
            final_chunks.append(current_text)
            overlap_text = current_text[-overlap:] if overlap > 0 else ""
            current_text = _normalize_whitespace(overlap_text + " " + chunk)
        else:
            current_text = combined
    if current_text:
        final_chunks.append(current_text)
    return final_chunks


# ── Semantic Chunking ─────────────────────────────────────────────────────────

def _split_into_sentences(text: str) -> list[str]:
    text = _normalize_whitespace(text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _semantic_split(
    text: str,
    embed_fn: Callable[[list[str]], np.ndarray],
    max_chars: int,
    similarity_threshold: float = 0.45,
    embed_batch_size: int = 128,
) -> list[str]:
    """Tách văn bản dựa trên cosine similarity — khi sim giảm đột ngột → chunk mới."""
    sentences = _split_into_sentences(text)
    if len(sentences) <= 2:
        return _split_long_text(text, max_chars)

    all_embeddings: list[np.ndarray] = []
    for i in range(0, len(sentences), embed_batch_size):
        batch = sentences[i:i + embed_batch_size]
        all_embeddings.append(embed_fn(batch))
    embeddings = np.concatenate(all_embeddings, axis=0)

    # E5 đã normalize → dot product = cosine similarity
    sims = np.sum(embeddings[:-1] * embeddings[1:], axis=1)

    breakpoints = [0]
    for i, sim in enumerate(sims):
        if sim < similarity_threshold:
            breakpoints.append(i + 1)
    breakpoints.append(len(sentences))

    chunks: list[str] = []
    for start, end in zip(breakpoints[:-1], breakpoints[1:]):
        chunk = " ".join(sentences[start:end]).strip()
        if not chunk:
            continue
        if len(chunk) > max_chars:
            chunks.extend(_split_long_text(chunk, max_chars))
        else:
            chunks.append(chunk)

    return chunks


# ── Core: Parent-Child Chunking ───────────────────────────────────────────────

def build_parent_child_chunks(
    documents: list,
    parent_chunk_size: int = 2000,
    parent_chunk_overlap: int = 400,
    child_chunk_size: int = 600,
    child_chunk_overlap: int = 200,
    id_registry_path: str | os.PathLike | None = None,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    similarity_threshold: float = 0.45,
) -> tuple[list, dict, list]:
    """Small-to-Big Retrieval với semantic chunking tùy chọn.

    embed_fn: hàm nhận list[str] và trả np.ndarray (n, dim) đã normalize.
              Nếu None, fallback về character-based splitting.

    Returns: (child_docs, parent_store, parent_docs)
    """
    child_docs, parent_docs, parent_store = [], [], {}
    id_registry = _load_chunk_id_registry(id_registry_path)
    use_semantic = embed_fn is not None

    for doc in documents:
        base_metadata = _to_json_safe_metadata(doc.metadata)

        p_chunks = _create_chunks_with_overlap(
            doc.page_content,
            chunk_size=parent_chunk_size,
            overlap=parent_chunk_overlap,
        )

        for parent_chunk_index, p_text in enumerate(p_chunks, start=1):
            parent_id = _assign_stable_chunk_id(
                id_registry,
                "parent",
                _build_chunk_signature_key({
                    "chunk_type": "parent",
                    "source": base_metadata.get("source"),
                    "section_title": base_metadata.get("section_title"),
                    "title": base_metadata.get("title"),
                    "chunk_index": parent_chunk_index,
                    "page_content": _normalize_whitespace(p_text),
                }),
            )

            parent_store[parent_id] = p_text
            p_metadata = deepcopy(base_metadata)
            p_metadata.update({"doc_id": parent_id, "parent_id": parent_id, "chunk_type": "parent"})
            p_doc = deepcopy(doc)
            p_doc.page_content = p_text
            p_doc.metadata = p_metadata
            parent_docs.append(p_doc)

            if use_semantic:
                c_chunks = _semantic_split(
                    p_text,
                    embed_fn=embed_fn,
                    max_chars=child_chunk_size,
                    similarity_threshold=similarity_threshold,
                )
            else:
                c_chunks = _create_chunks_with_overlap(
                    p_text,
                    chunk_size=child_chunk_size,
                    overlap=child_chunk_overlap,
                )

            for child_chunk_index, c_text in enumerate(c_chunks, start=1):
                child_id = _assign_stable_chunk_id(
                    id_registry,
                    "child",
                    _build_chunk_signature_key({
                        "chunk_type": "child",
                        "parent_id": parent_id,
                        "source": base_metadata.get("source"),
                        "section_title": base_metadata.get("section_title"),
                        "title": base_metadata.get("title"),
                        "chunk_index": child_chunk_index,
                        "page_content": _normalize_whitespace(c_text),
                    }),
                )
                c_metadata = deepcopy(p_metadata)
                c_metadata.update({"chunk_type": "child", "parent_id": parent_id, "child_id": child_id, "doc_id": child_id})
                c_doc = deepcopy(doc)
                c_doc.page_content = c_text
                c_doc.metadata = c_metadata
                child_docs.append(c_doc)

    _save_chunk_id_registry(id_registry, id_registry_path)
    mode = "semantic" if use_semantic else "character-based"
    print(f"[{mode}] Đã tạo {len(parent_docs)} parent chunks và {len(child_docs)} child chunks")

    return child_docs, parent_store, parent_docs


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_documents(documents: list, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = [
        {"page_content": doc.page_content, "metadata": _to_json_safe_metadata(doc.metadata)}
        for doc in documents
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu {len(payload)} documents vào {output_path}")


__all__ = [
    "build_parent_child_chunks",
    "clean_documents",
    "clean_text",
    "save_documents",
]
