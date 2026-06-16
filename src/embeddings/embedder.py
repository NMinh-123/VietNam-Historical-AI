"""Mô hình embedding E5 multilingual cho dense vector search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from src.app_config import get_config as _get_config

_emb_cfg = _get_config().model.embedding
E5_EMBEDDING_MODEL_NAME = _emb_cfg.name
E5_EMBEDDING_DIM = _emb_cfg.dim
E5_MAX_LENGTH = _emb_cfg.max_length

E5_QUERY_PROMPT_NAME = "query"
E5_PASSAGE_PROMPT_NAME = "passage"

E5_PROMPTS = {
    E5_QUERY_PROMPT_NAME: "query: ",
    E5_PASSAGE_PROMPT_NAME: "passage: ",
}


@dataclass(slots=True)
class E5EmbeddingConfig:
    model_name: str = E5_EMBEDDING_MODEL_NAME
    prompt_name: str = E5_PASSAGE_PROMPT_NAME
    batch_size: int = 32
    normalize_embeddings: bool = True
    device: str | None = None

    @property
    def embedding_dim(self) -> int:
        return E5_EMBEDDING_DIM


class E5EmbeddingModel:
    """Wrapper SentenceTransformer E5 với prompt tự động cho query/passage."""

    def __init__(self, config: E5EmbeddingConfig | None = None):
        self.config = config or E5EmbeddingConfig()
        self.model = SentenceTransformer(
            self.config.model_name,
            prompts=E5_PROMPTS,
            default_prompt_name=self.config.prompt_name,
            device=self.config.device,
        )

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    def embed(self, texts: list[str]) -> np.ndarray:
        """Nhúng danh sách văn bản, trả về float32 array đã normalize."""
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        encoded = self.model.encode(
            texts,
            prompt_name=self.config.prompt_name,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
        )
        return np.asarray(encoded, dtype=np.float32)


__all__ = [
    "E5EmbeddingConfig",
    "E5EmbeddingModel",
    "E5_EMBEDDING_DIM",
    "E5_EMBEDDING_MODEL_NAME",
    "E5_MAX_LENGTH",
    "E5_PASSAGE_PROMPT_NAME",
    "E5_PROMPTS",
    "E5_QUERY_PROMPT_NAME",
]
