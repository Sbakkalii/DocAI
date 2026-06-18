"""Retry helper for LLM calls with exponential backoff."""

import asyncio
import functools
import logging

logger = logging.getLogger("utils.retry")


async def retry_ollama(coro_factory, max_retries: int = 3, retry_delay: float = 2.0):
    """Execute a coroutine with retry and exponential backoff.

    Args:
        coro_factory: A zero-argument callable that returns a coroutine to execute.
        max_retries: Maximum number of attempts.
        retry_delay: Base delay in seconds between retries (doubles each attempt).

    Returns:
        The coroutine's return value.

    Raises:
        The last exception encountered after all retries are exhausted.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return await coro_factory()
        except Exception as e:
            last_error = e
            logger.warning(f"Call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                delay = retry_delay * (2 ** attempt)
                await asyncio.sleep(delay)
    raise last_error


def with_retry(max_retries: int = 3, retry_delay: float = 2.0):
    """Decorator: wrap an async function with exponential-backoff retry."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    if attempt < max_retries - 1:
                        delay = retry_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
            raise last_error
        return wrapper
    return decorator
