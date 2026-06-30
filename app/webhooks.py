"""
Webhook notification system for pipeline completion events.

Supports outbound HTTP POST to registered endpoints when pipeline
runs complete, enabling async integration with external systems.
"""

import asyncio
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

WEBHOOK_TIMEOUT = int(os.environ.get("WEBHOOK_TIMEOUT", "10"))
WEBHOOK_RETRIES = int(os.environ.get("WEBHOOK_RETRIES", "3"))

_registered_webhooks: list[str] = []


def register_webhook(url: str):
    """Register a webhook URL for pipeline completion notifications."""
    url = url.strip()
    if url and url not in _registered_webhooks:
        _registered_webhooks.append(url)
        logger.info(f"Webhook registered: {url}")


def get_webhooks() -> list[str]:
    return list(_registered_webhooks)


def clear_webhooks():
    _registered_webhooks.clear()


async def notify_pipeline_completed(
    session_id: str,
    status: str = "completed",
    metadata: dict[str, Any] | None = None,
):
    """Fire webhooks for pipeline completion."""
    if not _registered_webhooks:
        return

    payload = {
        "event": "pipeline_completed",
        "session_id": session_id,
        "status": status,
        "metadata": metadata or {},
    }

    async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT) as client:
        for url in _registered_webhooks:
            for attempt in range(WEBHOOK_RETRIES):
                try:
                    await client.post(url, json=payload)
                    logger.debug(f"Webhook delivered to {url}")
                    break
                except Exception as e:
                    if attempt == WEBHOOK_RETRIES - 1:
                        logger.warning(f"Webhook failed after {WEBHOOK_RETRIES} retries: {url} — {e}")
                    await asyncio.sleep(2 ** attempt)
