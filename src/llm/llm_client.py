"""LLM client: gọi mô hình ngôn ngữ với semaphore, rate limit và retry."""

from __future__ import annotations

import asyncio
import logging
import os
import re

_logger = logging.getLogger(__name__)

# ── Cấu hình mặc định ─────────────────────────────────────────────────────────

from src.app_config import get_config as _get_config

_lp_cfg = _get_config().llm_provider
_llm_cfg = _get_config().model.llm

DEFAULT_LLM_MODEL = _llm_cfg.default
DEFAULT_SHOPAIKEY_MODEL = _llm_cfg.shopaikey_default
DEFAULT_MAX_CONCURRENCY = _lp_cfg.max_concurrency
DEFAULT_MAX_RETRIES = _lp_cfg.max_retries
DEFAULT_RPM = 10
RETRY_DELAYS = _lp_cfg.retry_delays
RETRYABLE_STATUS_CODES = _lp_cfg.retryable_status_codes

# Giới hạn RPM theo prefix model
RPM_BY_MODEL_PREFIX: dict[str, int] = {
    "gemini-2.5-pro": 200,
    "gpt-5.4-mini": 200,
    "gpt-5-mini": 200,
}


# ── URL normalization ─────────────────────────────────────────────────────────

def _normalize_base_url(base_url: str) -> str:
    """Chuẩn hoá base URL về dạng .../v1, bỏ trailing /chat/completions."""
    normalized = (base_url or "").strip().rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    if normalized and not re.search(r"/v\d+$", normalized) and not normalized.endswith("/openai"):
        normalized = f"{normalized}/v1"
    return normalized


def _is_shopaikey(base_url: str) -> bool:
    return "api.shopaikey.com" in (base_url or "").lower()


LLM_BASE_URL: str = _normalize_base_url(
    os.getenv("OPENAI_COMPAT_BASE_URL")
    or os.getenv("SHOPAIKEY_BASE_URL")
    or "https://generativelanguage.googleapis.com/v1beta/openai/"
)


# ── Env helpers ───────────────────────────────────────────────────────────────

def require_api_key() -> str:
    """Đọc API key từ env; raise ValueError nếu không tìm thấy."""
    key = (
        os.getenv("GEMINI_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("SHOPAIKEY_TOKEN")
        or os.getenv("SHOPAIKEY_API_KEY")
    )
    if not key:
        raise ValueError(
            "Không tìm thấy API key. Hãy đặt GEMINI_KEY, OPENAI_API_KEY hoặc SHOPAIKEY_TOKEN."
        )
    return key


def resolve_model_name(model_name: str | None = None) -> str:
    """Đọc tên model từ tham số hoặc env; fallback về default theo provider."""
    resolved = (
        model_name
        or os.getenv("LIGHTRAG_MODEL_NAME")
        or os.getenv("SHOPAIKEY_MODEL_NAME")
        or os.getenv("GEMINI_MODEL_NAME")
        or os.getenv("OPENAI_MODEL")
    )
    if not resolved:
        return DEFAULT_SHOPAIKEY_MODEL if _is_shopaikey(LLM_BASE_URL) else DEFAULT_LLM_MODEL
    return resolved


def resolve_rpm(model_name: str, rpm: int | None = None) -> int:
    """Xác định giới hạn RPM theo model; ưu tiên tham số > env > lookup table > default."""
    if rpm is not None:
        return rpm
    env_val = os.getenv("GEMINI_RPM_LIMIT")
    if env_val:
        return int(env_val)
    for prefix, limit in RPM_BY_MODEL_PREFIX.items():
        if model_name.startswith(prefix):
            return limit
    return DEFAULT_RPM


# ── Rate Limiter ──────────────────────────────────────────────────────────────

class AsyncRequestRateLimiter:
    """Rate limiter token-bucket đơn giản cho async code."""

    def __init__(self, max_requests: int, window_seconds: float = 60.0):
        if max_requests <= 0:
            raise ValueError("max_requests phải lớn hơn 0")
        self.min_interval_seconds = window_seconds / max_requests
        self._next_available_at = 0.0
        self._lock: asyncio.Lock | None = None

    async def acquire(self) -> None:
        if self._lock is None:
            self._lock = asyncio.Lock()
        loop = asyncio.get_running_loop()
        async with self._lock:
            now = loop.time()
            scheduled_at = max(now, self._next_available_at)
            self._next_available_at = scheduled_at + self.min_interval_seconds
        sleep_for = scheduled_at - now
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)


# ── Retry helpers ─────────────────────────────────────────────────────────────

def _extract_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return status_code
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def _is_retryable(exc: Exception) -> bool:
    try:
        import httpx as _httpx
        if isinstance(exc, _httpx.TimeoutException):
            return True
    except ImportError:
        pass

    error_text = str(exc).lower()
    non_retryable = ("model_not_found", "invalid_model", "model not found")
    if any(m in error_text for m in non_retryable):
        return False
    if _extract_status_code(exc) in RETRYABLE_STATUS_CODES:
        return True
    retryable_markers = (
        "high demand", "unavailable", "temporarily unavailable",
        "try again later", "rate limit", "too many requests",
        "timeout", "timed out",
    )
    return any(m in error_text for m in retryable_markers)


# ── LLM Function Builder ──────────────────────────────────────────────────────

def build_llm_func(
    api_key: str,
    model_name: str,
    requests_per_minute: int,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_url: str | None = None,
):
    """Tạo async LLM function bọc openai_complete_if_cache với semaphore + rate limit + retry.

    Returns async callable với signature:
      async (prompt, system_prompt=None, history_messages=None, **kwargs) -> str
    """
    from lightrag.llm.openai import openai_complete_if_cache

    effective_base_url = base_url or LLM_BASE_URL
    rate_limiter = AsyncRequestRateLimiter(max_requests=requests_per_minute)
    semaphore: asyncio.Semaphore | None = None

    async def _llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        nonlocal semaphore
        if semaphore is None:
            semaphore = asyncio.Semaphore(max_concurrency)
        async with semaphore:
            for attempt in range(max_retries):
                await rate_limiter.acquire()
                try:
                    return await openai_complete_if_cache(
                        model_name,
                        prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages or [],
                        api_key=api_key,
                        base_url=effective_base_url,
                        **kwargs,
                    )
                except Exception as exc:
                    if attempt == max_retries - 1 or not _is_retryable(exc):
                        raise
                    delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                    _logger.warning(
                        "LLM retry %d/%d sau %.0fs (status=%s): %s",
                        attempt + 1, max_retries, delay, _extract_status_code(exc), exc,
                    )
                    await asyncio.sleep(delay)

    return _llm_func


__all__ = [
    "AsyncRequestRateLimiter",
    "DEFAULT_LLM_MODEL",
    "LLM_BASE_URL",
    "RETRY_DELAYS",
    "RETRYABLE_STATUS_CODES",
    "build_llm_func",
    "require_api_key",
    "resolve_model_name",
    "resolve_rpm",
]
