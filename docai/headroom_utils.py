"""
Headroom integration utilities for DocAI pipeline compression.

Wraps the headroom-ai library's compress() function for use in DocAI's
pipeline steps. Compresses JSON extraction results, RAG context, and
other content before it reaches the LLM to save context window space.

Usage:
    from docai.headroom_utils import compress_json, compress_text, is_available

    if is_available():
        compressed = compress_json(page_extractions)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

HEADROOM_AVAILABLE = False

try:
    from headroom import compress as _headroom_compress
    HEADROOM_AVAILABLE = True
except ImportError:
    logger.debug("headroom-ai not installed — compression disabled")


def is_available() -> bool:
    """Check if headroom-ai compression library is available."""
    return HEADROOM_AVAILABLE


def compress_content(
    content: str,
    target_ratio: float = 0.3,
    protect_recent: int = 0,
) -> str | None:
    """Compress raw text/JSON content using headroom's pipeline.

    Wraps content in a system message (allows compression), runs
    headroom's SmartCrusher (JSON) and Kompress-v2-base (text), and
    returns the compressed content string.

    Args:
        content: Raw text or JSON string to compress.
        target_ratio: Target compression ratio (0.0-1.0). Lower = more
                      aggressive compression. Default 0.3 means aim for
                      ~30% of original size.
        protect_recent: Number of recent messages to protect from
                        compression. 0 = compress everything.

    Returns:
        Compressed content string, or None if headroom unavailable.
    """
    if not HEADROOM_AVAILABLE:
        return None

    try:
        messages = [{"role": "system", "content": content}]
        result = _headroom_compress(
            messages=messages,
            protect_recent=protect_recent,
            target_ratio=target_ratio,
            compress_user_messages=True,
        )
        compressed = result.messages[0]["content"]
        if result.compression_ratio > 0:
            logger.debug(
                f"Headroom: {len(content)} -> {len(compressed)} chars "
                f"({result.compression_ratio:.1%}, "
                f"tx: {','.join(result.transforms_applied)})"
            )
        return compressed
    except Exception as e:
        logger.warning(f"Headroom compression failed: {e}")
        return content


def compress_json(
    data: Any,
    target_ratio: float = 0.3,
) -> str | None:
    """Compress JSON-serializable data using headroom's SmartCrusher.

    Args:
        data: Dict, list, or other JSON-serializable data.
        target_ratio: Target compression ratio (0.0-1.0).

    Returns:
        Compressed JSON string, or None if headroom unavailable.
    """
    import json

    if not HEADROOM_AVAILABLE:
        return None

    try:
        content = json.dumps(data, default=str)
        return compress_content(content, target_ratio=target_ratio)
    except Exception as e:
        logger.warning(f"Headroom JSON compression failed: {e}")
        return json.dumps(data, default=str)


def compress_text(
    text: str,
    target_ratio: float = 0.3,
) -> str | None:
    """Compress plain text using headroom's Kompress-v2-base model.

    Args:
        text: Plain text to compress.
        target_ratio: Target compression ratio (0.0-1.0).

    Returns:
        Compressed text string, or None if headroom unavailable.
    """
    if not HEADROOM_AVAILABLE:
        return None

    try:
        return compress_content(text, target_ratio=target_ratio)
    except Exception as e:
        logger.warning(f"Headroom text compression failed: {e}")
        return text


def compress_message_list(
    messages: list[dict[str, Any]],
    target_ratio: float = 0.3,
    protect_recent: int = 0,
) -> list[dict[str, Any]] | None:
    """Compress a list of OpenAI/Anthropic-format messages.

    This is a thin wrapper around headroom.compress() for multi-message
    compression (e.g., QA conversation context).

    Args:
        messages: List of message dicts with 'role' and 'content'.
        target_ratio: Target compression ratio.
        protect_recent: Number of recent messages to protect.

    Returns:
        Compressed message list, or None if headroom unavailable.
    """
    if not HEADROOM_AVAILABLE:
        return None

    try:
        result = _headroom_compress(
            messages=messages,
            protect_recent=protect_recent,
            target_ratio=target_ratio,
            compress_user_messages=True,
        )
        if result.compression_ratio > 0:
            logger.debug(
                f"Headroom messages: {len(messages)} msgs, "
                f"{result.compression_ratio:.1%} compression"
            )
        return result.messages
    except Exception as e:
        logger.warning(f"Headroom message compression failed: {e}")
        return messages
