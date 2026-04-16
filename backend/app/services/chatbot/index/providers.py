"""Các helper cho provider LLM kiểu OpenAI-compatible."""

from __future__ import annotations

import asyncio
import logging

_logger = logging.getLogger(__name__)

from .config import (
    GEMINI_OPENAI_BASE_URL,
    GEMINI_RETRYABLE_STATUS_CODES,
    GEMINI_TRANSIENT_RETRY_DELAYS_SECONDS,
)
from .runtime import openai_complete_if_cache


def _extract_exception_status_code(exc: Exception) -> int | None:
    """Rút status code từ exception theo các shape response phổ biến."""
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return status_code

    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def _is_retryable_gemini_exception(exc: Exception) -> bool:
    """Các lỗi hạ tầng thường nên retry thay vì fail ngay."""
    status_code = _extract_exception_status_code(exc)
    if status_code in GEMINI_RETRYABLE_STATUS_CODES:
        return True

    error_text = str(exc).lower()
    retryable_markers = (
        "high demand",
        "unavailable",
        "temporarily unavailable",
        "try again later",
        "rate limit",
        "too many requests",
    )
    return any(marker in error_text for marker in retryable_markers)


class AsyncRequestRateLimiter:
    """Rate limiter đơn giản để không dồn toàn bộ request vào provider."""

    def __init__(self, max_requests: int, window_seconds: float = 60.0):
        """Khởi tạo limiter dựa trên số request tối đa trong một cửa sổ thời gian."""
        if max_requests <= 0:
            raise ValueError("max_requests phải lớn hơn 0")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.min_interval_seconds = window_seconds / max_requests
        self._next_available_at = 0.0
        self._lock: asyncio.Lock | None = None  # lazy-init để tránh tạo ngoài event loop

    async def acquire(self) -> None:
        """Chờ tới thời điểm request kế tiếp được phép chạy."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        loop = asyncio.get_running_loop()
        async with self._lock:
            now = loop.time()
            scheduled_at = max(now, self._next_available_at)
            self._next_available_at = (
                scheduled_at + self.min_interval_seconds
            )

        sleep_for = scheduled_at - now
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)


def _build_gemini_llm_func(
    gemini_key: str,
    gemini_model_name: str,
    requests_per_minute: int,
    max_concurrency: int,
    transient_max_retries: int,
):
    """Bọc lời gọi LLM bằng semaphore + retry + rate limit."""
    request_limiter = AsyncRequestRateLimiter(
        max_requests=requests_per_minute,
        window_seconds=60.0,
    )
    request_semaphore: asyncio.Semaphore | None = None  # lazy-init để tránh tạo ngoài event loop

    async def _gemini_llm_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        **kwargs,
    ):
        """Gọi provider qua lớp bảo vệ concurrency, rate limit và retry lỗi tạm thời."""
        nonlocal request_semaphore
        if request_semaphore is None:
            request_semaphore = asyncio.Semaphore(max_concurrency)
        async with request_semaphore:
            for attempt_index in range(transient_max_retries):
                await request_limiter.acquire()
                try:
                    return await openai_complete_if_cache(
                        gemini_model_name,
                        prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages or [],
                        api_key=gemini_key,
                        base_url=GEMINI_OPENAI_BASE_URL,
                        **kwargs,
                    )
                except Exception as exc:
                    is_last_attempt = (
                        attempt_index == transient_max_retries - 1
                    )
                    if is_last_attempt or not _is_retryable_gemini_exception(exc):
                        raise

                    retry_delay_seconds = GEMINI_TRANSIENT_RETRY_DELAYS_SECONDS[
                        min(
                            attempt_index,
                            len(GEMINI_TRANSIENT_RETRY_DELAYS_SECONDS) - 1,
                        )
                    ]
                    status_code = _extract_exception_status_code(exc)
                    _logger.warning(
                        "LLM provider tạm quá tải hoặc chạm retryable error "
                        "(status=%s, attempt=%d/%d). Chờ %.0fs rồi thử lại...",
                        status_code, attempt_index + 1, transient_max_retries, retry_delay_seconds,
                    )
                    await asyncio.sleep(retry_delay_seconds)

    return _gemini_llm_func


__all__ = [
    "AsyncRequestRateLimiter",
    "_build_gemini_llm_func",
    "_extract_exception_status_code",
    "_is_retryable_gemini_exception",
]
