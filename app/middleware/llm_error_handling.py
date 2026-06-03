"""M1: LLM Error Handling Middleware

Provides retry with exponential backoff and circuit breaker for LLM API calls.

Integration points:
  - react_nodes.create_model_node  →  replaces llm.invoke()
  - supervisor_plan_execute.planner_node  →  replaces llm.invoke()
"""

import time
import random
from typing import List, Any, Optional

from langchain_core.messages import BaseMessage

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Status codes that are safe to retry
_RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}
_AUTH_CODES = {401, 403}


# ── Errors ──────────────────────────────────────────────────────────

class LLMCircuitBreakerError(Exception):
    """Circuit breaker is open – reject the request."""


class LLMMaxRetriesError(Exception):
    """All retry attempts exhausted."""


# ── Circuit Breaker ─────────────────────────────────────────────────

class CircuitBreaker:
    """Prevents cascading failures by blocking calls after N consecutive failures."""

    def __init__(self, failure_threshold: int = 5, reset_time: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_time = reset_time
        self.failure_count = 0
        self.last_failure_time: float = 0
        self.state = "closed"  # closed | open | half-open

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                f"Circuit breaker OPEN after {self.failure_count} consecutive failures"
            )

    def record_success(self):
        self.failure_count = 0
        self.state = "closed"

    def can_execute(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure_time >= self.reset_time:
                self.state = "half-open"
                logger.info("Circuit breaker → HALF-OPEN, allowing test request")
                return True
            return False
        return True  # half-open


# ── Error classifier ────────────────────────────────────────────────

def _classify_error(error: Exception) -> dict:
    """Return ``{"type": str, "retryable": bool}`` for the given exception."""
    s = str(error).lower()

    for code in _RETRYABLE_CODES:
        if str(code) in s:
            return {"type": f"http_{code}", "retryable": True}
    for code in _AUTH_CODES:
        if str(code) in s:
            return {"type": f"http_{code}", "retryable": False}

    if "rate limit" in s or "too many requests" in s:
        return {"type": "rate_limit", "retryable": True}
    if "timeout" in s or "timed out" in s:
        return {"type": "timeout", "retryable": True}
    if "quota" in s:
        return {"type": "quota", "retryable": False}
    if "auth" in s or "api key" in s or "unauthorized" in s:
        return {"type": "auth", "retryable": False}
    if "server error" in s or "internal server" in s:
        return {"type": "server_error", "retryable": True}
    if "connection" in s:
        return {"type": "connection", "retryable": True}

    # Unknown errors: retry once by default
    return {"type": "unknown", "retryable": True}


# ── Main handler ────────────────────────────────────────────────────

class LLMRetryHandler:
    """Wraps ``llm.invoke()`` with exponential-backoff retry and circuit breaker."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_reset: float = 60.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            reset_time=circuit_breaker_reset,
        )

    def invoke(self, llm, messages: List, **kwargs) -> Any:
        """Call *llm.invoke(messages)* with retry + circuit-breaker protection."""
        if not self.circuit_breaker.can_execute():
            raise LLMCircuitBreakerError(
                f"LLM 熔断器已开启（连续失败 {self.circuit_breaker.failure_count} 次），"
                f"请 {self.circuit_breaker.reset_time:.0f}s 后重试。"
            )

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = llm.invoke(messages, **kwargs)
                self.circuit_breaker.record_success()
                return response
            except Exception as e:
                last_error = e
                info = _classify_error(e)

                if not info["retryable"]:
                    logger.error(f"LLM non-retryable error ({info['type']}): {e}")
                    self.circuit_breaker.record_failure()
                    raise

                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (2 ** attempt) + random.uniform(0, 1),
                        self.max_delay,
                    )
                    logger.warning(
                        f"LLM retryable error ({info['type']}) "
                        f"attempt {attempt + 1}/{self.max_retries + 1}: {e}. "
                        f"Retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
                else:
                    self.circuit_breaker.record_failure()
                    raise LLMMaxRetriesError(
                        f"LLM 调用在 {self.max_retries} 次重试后仍然失败: {e}"
                    ) from e

        # Should never reach here, but satisfy type checkers
        raise last_error  # type: ignore[misc]


# ── Singleton ───────────────────────────────────────────────────────

_retry_handler: Optional[LLMRetryHandler] = None


def get_llm_retry_handler() -> LLMRetryHandler:
    """Return (and lazily create) the global retry handler."""
    global _retry_handler
    if _retry_handler is None:
        from app.config.settings import settings
        _retry_handler = LLMRetryHandler(
            max_retries=getattr(settings, "LLM_MAX_RETRIES", 3),
            base_delay=getattr(settings, "LLM_RETRY_BASE_DELAY", 1.0),
            circuit_breaker_threshold=getattr(settings, "LLM_CIRCUIT_BREAKER_THRESHOLD", 5),
            circuit_breaker_reset=getattr(settings, "LLM_CIRCUIT_BREAKER_RESET_TIME", 60.0),
        )
    return _retry_handler
