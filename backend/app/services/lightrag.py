"""Facade giữ API cũ cho phần còn lại của backend.

File này chỉ re-export các thành phần từ `services/index/` để:
- `query_engine.py` không phải đổi import,
- logic LightRAG được chia nhỏ ra nhiều module dễ bảo trì hơn.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


# Khi chạy file này trực tiếp bằng `python services/lightrag.py`,
# cần đưa thư mục `app` vào sys.path để import package `services` ổn định.
CURRENT_DIR = Path(__file__).resolve().parent
APP_DIR = CURRENT_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from services.index import (  # noqa: E402
    CHILD_DOCSTORE_PATH,
    COLLECTION_NAME,
    DATA_DIR,
    DEFAULT_GEMINI_MODEL_NAME,
    DENSE_MODEL_NAME,
    DocStatus,
    EmbeddingFunc,
    GEMINI_OPENAI_BASE_URL,
    LIGHTRAG_INGEST_MANIFEST_PATH,
    LIGHTRAG_WORKSPACE,
    LightRAG,
    PARENT_DOCSTORE_PATH,
    QDRANT_DB_PATH,
    RAW_DATA_PATH,
    SPARSE_MODEL_NAME,
    _build_gemini_llm_func,
    _read_env_flag,
    _read_env_int,
    _read_env_str,
    _require_gemini_key,
    _resolve_gemini_max_concurrency,
    _resolve_gemini_model_name,
    _resolve_gemini_rpm_limit,
    _resolve_gemini_transient_max_retries,
    _resolve_lightrag_batch_size,
    _resolve_lightrag_max_parallel_insert,
    _resolve_resume_existing_queue,
    compute_mdhash_id,
    hybrid_ingest,
    openai_complete_if_cache,
    sanitize_text_for_encoding,
)
from services.reset_outputs import reset_generated_outputs  # noqa: E402

__all__ = [
    "CHILD_DOCSTORE_PATH",
    "COLLECTION_NAME",
    "CURRENT_DIR",
    "DATA_DIR",
    "DEFAULT_GEMINI_MODEL_NAME",
    "DENSE_MODEL_NAME",
    "DocStatus",
    "EmbeddingFunc",
    "GEMINI_OPENAI_BASE_URL",
    "LIGHTRAG_INGEST_MANIFEST_PATH",
    "LIGHTRAG_WORKSPACE",
    "LightRAG",
    "PARENT_DOCSTORE_PATH",
    "QDRANT_DB_PATH",
    "RAW_DATA_PATH",
    "SPARSE_MODEL_NAME",
    "_build_gemini_llm_func",
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
    "_resolve_resume_existing_queue",
    "compute_mdhash_id",
    "hybrid_ingest",
    "openai_complete_if_cache",
    "sanitize_text_for_encoding",
]


if __name__ == "__main__":
    if _read_env_flag("LIGHTRAG_RESET_OUTPUTS", default=False):
        cleanup_results = reset_generated_outputs(
            remove_id_registry=_read_env_flag(
                "LIGHTRAG_RESET_ID_REGISTRY",
                default=False,
            ),
            drop_qdrant_collection=_read_env_flag(
                "LIGHTRAG_DROP_QDRANT_COLLECTION",
                default=False,
            ),
            qdrant_host=_read_env_str("QDRANT_HOST") or "localhost",
            qdrant_port=_read_env_int("QDRANT_PORT") or 6333,
            qdrant_collection=_read_env_str("QDRANT_COLLECTION_NAME")
            or COLLECTION_NAME,
        )
        print("Đã dọn artefact cũ:", cleanup_results)

    asyncio.run(
        hybrid_ingest(
            test_mode=_read_env_flag("LIGHTRAG_TEST_MODE", default=False),
            recreate_collection=_read_env_flag(
                "LIGHTRAG_RECREATE_QDRANT",
                default=False,
            ),
        )
    )
