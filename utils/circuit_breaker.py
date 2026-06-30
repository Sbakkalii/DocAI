"""
Circuit breaker for external service calls (Ollama, vLLM).

Prevents cascading failures when a downstream service is slow or unavailable.
After N consecutive failures, the breaker opens and fails fast instead of
retrying, allowing the service to recover.
"""

import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Fail-fast circuit breaker for async service calls.

    States: CLOSED (normal) → OPEN (after threshold failures) → HALF_OPEN (after timeout)
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._failure_count = 0
        self._last_failure_time = 0.0
        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._half_open_calls = 0

    @property
    def state(self) -> str:
        return self._state

    async def call(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Call the function with circuit breaker protection."""
        if self._state == "OPEN":
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = "HALF_OPEN"
                self._half_open_calls = 0
                logger.info(f"Circuit {self.name}: OPEN → HALF_OPEN")
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit {self.name} is OPEN — "
                    f"failing fast ({self._failure_count} failures)"
                )

        if self._state == "HALF_OPEN":
            if self._half_open_calls >= self.half_open_max_calls:
                raise CircuitBreakerOpenError(
                    f"Circuit {self.name} is HALF_OPEN — too many probes"
                )
            self._half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise

    def _on_success(self):
        if self._state == "HALF_OPEN":
            logger.info(f"Circuit {self.name}: HALF_OPEN → CLOSED (probe succeeded)")
        self._failure_count = 0
        self._state = "CLOSED"

    def _on_failure(self, error: Exception):
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            self._state = "OPEN"
            logger.warning(
                f"Circuit {self.name}: CLOSED → OPEN "
                f"({self._failure_count} failures, reason: {error})"
            )


class CircuitBreakerOpenError(Exception):
    """Raised when a circuit breaker is open and blocking calls."""
    pass


# Global circuit breakers
_ollama_breaker: CircuitBreaker | None = None
_vllm_breaker: CircuitBreaker | None = None


def get_ollama_breaker() -> CircuitBreaker:
    global _ollama_breaker
    if _ollama_breaker is None:
        _ollama_breaker = CircuitBreaker(name="ollama")
    return _ollama_breaker


def get_vllm_breaker() -> CircuitBreaker:
    global _vllm_breaker
    if _vllm_breaker is None:
        _vllm_breaker = CircuitBreaker(name="vllm")
    return _vllm_breaker
