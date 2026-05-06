# Cấu hình đường dẫn, model và biến môi trường cho toàn bộ pipeline.

from __future__ import annotations

import os
import re
from pathlib import Path

from data.process_data.e5_embeddings import E5_EMBEDDING_MODEL_NAME

# Load .env từ gốc dự án (nếu có) — không ghi đè biến đã tồn tại trong shell
def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[4] / ".env"
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

CURRENT_DIR = Path(__file__).resolve().parent  # services/chatbot/index_and_retrieve/
APP_DIR = CURRENT_DIR.parent.parent.parent      # app/
DATA_DIR = APP_DIR / "data"

RAW_DATA_PATH = DATA_DIR / "ocr_data"
QDRANT_DB_PATH = DATA_DIR / "qdrant_db"
LIGHTRAG_WORKSPACE = DATA_DIR / "lightrag_storage"
LIGHTRAG_INGEST_MANIFEST_PATH = DATA_DIR / "lightrag_ingest_manifest.json"
PARENT_DOCSTORE_PATH = DATA_DIR / "parent_docs.json"
CHILD_DOCSTORE_PATH = DATA_DIR / "child_docs.json"

COLLECTION_NAME = "vietnam_history_hybrid"
DENSE_MODEL_NAME = E5_EMBEDDING_MODEL_NAME
SPARSE_MODEL_NAME = "Qdrant/bm25"

# Qdrant server — đọc từ env để dùng được cả Docker lẫn local
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
DEFAULT_GEMINI_MODEL_NAME = "gemini-3.1-flash-lite"
RPM_BY_MODEL_PREFIX = {
    "gemini-2.5-pro": 200,
    "gpt-5.4-mini": 200,
    "gpt-5-mini": 200,
}
DEFAULT_GEMINI_MAX_CONCURRENCY = 5
DEFAULT_GEMINI_TRANSIENT_MAX_RETRIES = 4
GEMINI_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
GEMINI_TRANSIENT_RETRY_DELAYS_SECONDS = (15.0, 30.0, 60.0, 120.0)
DEFAULT_LIGHTRAG_BATCH_SIZE = 20
DEFAULT_LIGHTRAG_MAX_PARALLEL_INSERT = 20
DEFAULT_QDRANT_BATCH_SIZE = 256
DEFAULT_SHOPAIKEY_MODEL_NAME = "claude-haiku-4-5"


def _normalize_openai_compatible_base_url(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    if normalized and not re.search(r"/v\d+$", normalized) and not normalized.endswith("/openai"):
        normalized = f"{normalized}/v1"
    return normalized


def _is_shopaikey_base_url(base_url: str) -> bool:
    return "api.shopaikey.com" in (base_url or "").lower()


GEMINI_OPENAI_BASE_URL = _normalize_openai_compatible_base_url(
    os.getenv("OPENAI_COMPAT_BASE_URL")
    or os.getenv("SHOPAIKEY_BASE_URL")
    or "https://generativelanguage.googleapis.com/v1beta/openai/"
)


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
    key = (
        gemini_key
        or os.getenv("GEMINI_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("SHOPAIKEY_TOKEN")
        or os.getenv("SHOPAIKEY_API_KEY")
    )
    if not key:
        raise ValueError(
            "KHÔNG TÌM THẤY API token trong môi trường. "
            "Hãy đặt GEMINI_KEY, OPENAI_API_KEY hoặc SHOPAIKEY_TOKEN."
        )
    return key


def _resolve_gemini_model_name(
    gemini_model_name: str | None = None,
) -> str:
    resolved_model_name = (
        gemini_model_name
        or _read_env_str("LIGHTRAG_MODEL_NAME")
        or _read_env_str("SHOPAIKEY_MODEL_NAME")
        or _read_env_str("GEMINI_MODEL_NAME")
        or _read_env_str("OPENAI_MODEL")
    )

    if not resolved_model_name:
        if _is_shopaikey_base_url(GEMINI_OPENAI_BASE_URL):
            return DEFAULT_SHOPAIKEY_MODEL_NAME
        return DEFAULT_GEMINI_MODEL_NAME

    return resolved_model_name


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
    "GEMINI_RETRYABLE_STATUS_CODES",
    "GEMINI_TRANSIENT_RETRY_DELAYS_SECONDS",
    "LIGHTRAG_INGEST_MANIFEST_PATH",
    "LIGHTRAG_WORKSPACE",
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
