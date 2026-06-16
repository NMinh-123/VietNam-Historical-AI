# Cấu hình đường dẫn, model và biến môi trường cho toàn bộ pipeline.

from __future__ import annotations

import os
from pathlib import Path

from app.core.app_config import get_config as _get_config
from app.core.embeddings.embedder import E5_EMBEDDING_MODEL_NAME
from app.core.llm.llm_client import (
    LLM_BASE_URL as GEMINI_OPENAI_BASE_URL,
    RPM_BY_MODEL_PREFIX,
    _is_shopaikey as _is_shopaikey_base_url,
    require_api_key as _require_api_key_base,
    resolve_model_name as _resolve_model_name_base,
)

_cfg = _get_config()

# Load .env từ gốc dự án (nếu có) — không ghi đè biến đã tồn tại trong shell
def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value

_load_dotenv()

CURRENT_DIR = Path(__file__).resolve().parent  # app/chatbot/rag/
DATA_DIR = Path(os.getenv("DATA_DIR") or (CURRENT_DIR.parent.parent.parent / "app" / "data"))

RAW_DATA_PATH = DATA_DIR / "ocr_data"
QDRANT_DB_PATH = DATA_DIR / "qdrant_db"
LIGHTRAG_WORKSPACE = DATA_DIR / "lightrag_storage"
LIGHTRAG_INGEST_MANIFEST_PATH = DATA_DIR / "lightrag_ingest_manifest.json"
PARENT_DOCSTORE_PATH = DATA_DIR / "parent_docs.json"
CHILD_DOCSTORE_PATH = DATA_DIR / "child_docs.json"

COLLECTION_NAME = _cfg.vectordb.collection_name
PARENT_COLLECTION_NAME = _cfg.vectordb.parent_collection_name
DENSE_MODEL_NAME = E5_EMBEDDING_MODEL_NAME
SPARSE_MODEL_NAME = _cfg.model.embedding.sparse

# Qdrant server — env var переопределяет config.yaml для Docker/prod
QDRANT_HOST = os.getenv("QDRANT_HOST") or _cfg.vectordb.host
QDRANT_PORT = int(os.getenv("QDRANT_PORT") or _cfg.vectordb.port)
DEFAULT_GEMINI_MODEL_NAME = _cfg.model.llm.default
DEFAULT_GEMINI_MAX_CONCURRENCY = _cfg.llm_provider.max_concurrency
DEFAULT_GEMINI_TRANSIENT_MAX_RETRIES = _cfg.llm_provider.max_retries
DEFAULT_LIGHTRAG_BATCH_SIZE = 20
DEFAULT_LIGHTRAG_MAX_PARALLEL_INSERT = 20
DEFAULT_QDRANT_BATCH_SIZE = _cfg.vectordb.batch_size
DEFAULT_SHOPAIKEY_MODEL_NAME = _cfg.model.llm.shopaikey_default


def _read_env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _read_env_int(name: str) -> int | None:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return None
    return int(raw_value.strip())


def _read_env_str(name: str) -> str | None:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return None
    return raw_value.strip()


def _require_gemini_key(gemini_key: str | None = None) -> str:
    if gemini_key:
        return gemini_key
    return _require_api_key_base()


def _resolve_gemini_model_name(gemini_model_name: str | None = None) -> str:
    return _resolve_model_name_base(gemini_model_name)


def _resolve_gemini_rpm_limit(
    gemini_model_name: str,
    requests_per_minute: int | None = None,
) -> int:
    if requests_per_minute is not None:
        return requests_per_minute

    env_override = _read_env_int("GEMINI_RPM_LIMIT")
    if env_override is not None:
        return env_override

    for model_prefix, rpm_limit in RPM_BY_MODEL_PREFIX.items():
        if gemini_model_name.startswith(model_prefix):
            return rpm_limit

    return 10


def _resolve_gemini_max_concurrency(
    max_concurrency: int | None = None,
) -> int:
    if max_concurrency is not None:
        return max_concurrency

    env_override = _read_env_int("GEMINI_MAX_CONCURRENCY")
    if env_override is not None:
        return env_override

    return DEFAULT_GEMINI_MAX_CONCURRENCY


def _resolve_gemini_transient_max_retries(
    max_retries: int | None = None,
) -> int:
    if max_retries is not None:
        return max_retries

    env_override = _read_env_int("GEMINI_TRANSIENT_MAX_RETRIES")
    if env_override is not None:
        return env_override

    return DEFAULT_GEMINI_TRANSIENT_MAX_RETRIES


def _resolve_qdrant_batch_size(batch_size: int | None = None) -> int:
    if batch_size is None:
        batch_size = _read_env_int("QDRANT_BATCH_SIZE")
    resolved = batch_size or DEFAULT_QDRANT_BATCH_SIZE
    return max(1, min(resolved, 1024))


def _resolve_lightrag_batch_size(batch_size: int | None = None) -> int:
    if batch_size is None:
        batch_size = _read_env_int("LIGHTRAG_BATCH_SIZE")
    resolved = batch_size or DEFAULT_LIGHTRAG_BATCH_SIZE
    return max(1, min(resolved, 10))


def _resolve_lightrag_max_parallel_insert(
    max_parallel_insert: int | None = None,
) -> int:
    if max_parallel_insert is None:
        max_parallel_insert = _read_env_int("LIGHTRAG_MAX_PARALLEL_INSERT")
    resolved = max_parallel_insert or DEFAULT_LIGHTRAG_MAX_PARALLEL_INSERT
    return max(1, min(resolved, 10))


def _resolve_resume_existing_queue(
    resume_existing_queue: bool | None = None,
) -> bool:
    if resume_existing_queue is not None:
        return resume_existing_queue
    return _read_env_flag("LIGHTRAG_RESUME_EXISTING_QUEUE", default=True)


__all__ = [
    "CHILD_DOCSTORE_PATH",
    "COLLECTION_NAME",
    "CURRENT_DIR",
    "DATA_DIR",
    "DEFAULT_GEMINI_MODEL_NAME",
    "DEFAULT_QDRANT_BATCH_SIZE",
    "DENSE_MODEL_NAME",
    "GEMINI_OPENAI_BASE_URL",
    "LIGHTRAG_INGEST_MANIFEST_PATH",
    "LIGHTRAG_WORKSPACE",
    "PARENT_COLLECTION_NAME",
    "PARENT_DOCSTORE_PATH",
    "QDRANT_DB_PATH",
    "QDRANT_HOST",
    "QDRANT_PORT",
    "RAW_DATA_PATH",
    "RPM_BY_MODEL_PREFIX",
    "SPARSE_MODEL_NAME",
    "_read_env_flag",
    "_read_env_int",
    "_read_env_str",
    "_require_gemini_key",
    "_resolve_gemini_max_concurrency",
    "_resolve_gemini_model_name",
    "_resolve_gemini_rpm_limit",
    "_resolve_gemini_transient_max_retries",
    "_resolve_lightrag_batch_size",
    "_resolve_lightrag_max_parallel_insert",
    "_resolve_qdrant_batch_size",
    "_resolve_resume_existing_queue",
]
