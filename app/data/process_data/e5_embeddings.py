from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

E5_EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
E5_QUERY_PROMPT_NAME = "query"
E5_PASSAGE_PROMPT_NAME = "passage"
E5_EMBEDDING_DIM = 384
E5_MAX_LENGTH = 512

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
    """Wrapper nhỏ cho multilingual-e5-small với prompt chuẩn query/passage."""

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
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        encoded = self.model.encode(
            texts,
            prompt_name=self.config.prompt_name,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
        )
        return np.asarray(encoded, dtype=np.float32)
