"""Đọc config.yaml một lần và expose dưới dạng singleton."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_YAML_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


@dataclass
class EmbeddingConfig:
    name: str
    sparse: str
    dim: int
    max_length: int


@dataclass
class LLMConfig:
    default: str
    shopaikey_default: str


@dataclass
class ModelConfig:
    embedding: EmbeddingConfig
    llm: LLMConfig


@dataclass
class RetrievalConfig:
    top_k: int
    limit: int
    broad_top_k: int
    broad_graph_top_k: int
    fused_weight: float
    lexical_weight: float
    count_bonus: float
    max_chunks_per_parent: int
    topic_shift_threshold: float


@dataclass
class ChunkingConfig:
    parent_chunk_size: int
    parent_chunk_overlap: int
    child_chunk_size: int
    child_chunk_overlap: int
    similarity_threshold: float
    embed_batch_size: int


@dataclass
class VectorDBConfig:
    collection_name: str
    parent_collection_name: str
    batch_size: int
    host: str
    port: int


@dataclass
class LLMProviderConfig:
    requests_per_minute: int
    max_concurrency: int
    max_retries: int
    retry_delays: tuple[float, ...]
    retryable_status_codes: frozenset[int]


@dataclass
class LightragConfig:
    chunk_token_size: int
    chunk_overlap_token_size: int
    batch_size: int
    max_parallel_insert: int


@dataclass
class HistoryConfig:
    summarize_threshold: int
    recency_turns: int


@dataclass
class DataConfig:
    raw_data_path: str
    lightrag_workspace: str
    qdrant_db_path: str
    parent_docstore: str
    ingest_manifest: str


@dataclass
class AppConfig:
    model: ModelConfig
    retrieval: RetrievalConfig
    chunking: ChunkingConfig
    vectordb: VectorDBConfig
    llm_provider: LLMProviderConfig
    lightrag: LightragConfig
    history: HistoryConfig
    data: DataConfig


_config: AppConfig | None = None


def get_config(yaml_path: Path | None = None) -> AppConfig:
    """Trả về AppConfig đã cache; đọc config.yaml nếu chưa load."""
    global _config
    if _config is None or yaml_path is not None:
        _config = _load(yaml_path or _YAML_PATH)
    return _config


def _load(path: Path) -> AppConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    m = raw["model"]
    r = raw["retrieval"]
    c = raw["chunking"]
    v = raw["vectordb"]
    lp = raw["llm_provider"]
    lg = raw["lightrag"]
    h = raw["history"]
    d = raw["data"]

    return AppConfig(
        model=ModelConfig(
            embedding=EmbeddingConfig(
                name=m["embedding"]["name"],
                sparse=m["embedding"]["sparse"],
                dim=m["embedding"]["dim"],
                max_length=m["embedding"]["max_length"],
            ),
            llm=LLMConfig(
                default=m["llm"]["default"],
                shopaikey_default=m["llm"]["shopaikey_default"],
            ),
        ),
        retrieval=RetrievalConfig(
            top_k=r["top_k"],
            limit=r["limit"],
            broad_top_k=r["broad_top_k"],
            broad_graph_top_k=r["broad_graph_top_k"],
            fused_weight=r["fused_weight"],
            lexical_weight=r["lexical_weight"],
            count_bonus=r["count_bonus"],
            max_chunks_per_parent=r["max_chunks_per_parent"],
            topic_shift_threshold=r["topic_shift_threshold"],
        ),
        chunking=ChunkingConfig(
            parent_chunk_size=c["parent_chunk_size"],
            parent_chunk_overlap=c["parent_chunk_overlap"],
            child_chunk_size=c["child_chunk_size"],
            child_chunk_overlap=c["child_chunk_overlap"],
            similarity_threshold=c["similarity_threshold"],
            embed_batch_size=c["embed_batch_size"],
        ),
        vectordb=VectorDBConfig(
            collection_name=v["collection_name"],
            parent_collection_name=v["parent_collection_name"],
            batch_size=v["batch_size"],
            host=v["host"],
            port=v["port"],
        ),
        llm_provider=LLMProviderConfig(
            requests_per_minute=lp["requests_per_minute"],
            max_concurrency=lp["max_concurrency"],
            max_retries=lp["max_retries"],
            retry_delays=tuple(float(x) for x in lp["retry_delays"]),
            retryable_status_codes=frozenset(lp["retryable_status_codes"]),
        ),
        lightrag=LightragConfig(
            chunk_token_size=lg["chunk_token_size"],
            chunk_overlap_token_size=lg["chunk_overlap_token_size"],
            batch_size=lg["batch_size"],
            max_parallel_insert=lg["max_parallel_insert"],
        ),
        history=HistoryConfig(
            summarize_threshold=h["summarize_threshold"],
            recency_turns=h["recency_turns"],
        ),
        data=DataConfig(
            raw_data_path=d["raw_data_path"],
            lightrag_workspace=d["lightrag_workspace"],
            qdrant_db_path=d["qdrant_db_path"],
            parent_docstore=d["parent_docstore"],
            ingest_manifest=d["ingest_manifest"],
        ),
    )


__all__ = [
    "AppConfig",
    "ChunkingConfig",
    "DataConfig",
    "EmbeddingConfig",
    "HistoryConfig",
    "LightragConfig",
    "LLMConfig",
    "LLMProviderConfig",
    "ModelConfig",
    "RetrievalConfig",
    "VectorDBConfig",
    "get_config",
]
