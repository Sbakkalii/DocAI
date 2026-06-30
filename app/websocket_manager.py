import asyncio
import contextlib
import logging
import time
from typing import Any

from fastapi import WebSocket


def _json_safe(obj: Any) -> Any:
    """Recursively convert non-serializable objects (e.g. Pydantic models) to dicts for JSON"""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):  # Pydantic v1
        return obj.dict()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    return obj


class WebSocketManager:
    def __init__(self):
        self._connections: dict[str, set[WebSocket]] = {}
        self._session_states: dict[str, list[dict]] = {}
        self._last_broadcast: dict[str, float] = {}
        self.logger = logging.getLogger("app.ws")

    async def connect(self, session_id: str, ws: WebSocket):
        await ws.accept()
        if session_id not in self._connections:
            self._connections[session_id] = set()
        self._connections[session_id].add(ws)

        # Replay all cached progress messages so far
        cached = self._session_states.get(session_id, [])
        for msg in cached:
            try:
                await ws.send_json(msg)
            except Exception:
                self.logger.debug(f"WS replay failed for session {session_id}, stopping replay")
                break

        self.logger.info(
            f"WS connected for session {session_id} "
            f"({len(self._connections[session_id])} clients, {len(cached)} cached events)"
        )

    async def disconnect(self, session_id: str, ws: WebSocket):
        if session_id in self._connections:
            self._connections[session_id].discard(ws)
            if not self._connections[session_id]:
                del self._connections[session_id]
                # Keep cached state around briefly in case of reconnect
                asyncio.ensure_future(self._clear_state_delayed(session_id))

    async def _clear_state_delayed(self, session_id: str, delay: float = 60.0):
        await asyncio.sleep(delay)
        self._session_states.pop(session_id, None)

    async def broadcast(self, session_id: str, message: dict[str, Any]):
        # Sanitise message so JSON serialisation never fails on e.g. Pydantic models
        safe = _json_safe(message)
        # Cache for late-connecting clients
        if session_id not in self._session_states:
            self._session_states[session_id] = []
        self._session_states[session_id].append(safe)

        if session_id not in self._connections:
            self._last_broadcast[session_id] = time.time()
            return

        # Idle heartbeat: if >25s since last broadcast, send a ping first
        now = time.time()
        last = self._last_broadcast.get(session_id, 0)
        if now - last > 25:
            for ws in list(self._connections[session_id]):
                with contextlib.suppress(Exception):
                    await ws.send_json({"type": "ping"})

        dead = set()
        for ws in self._connections[session_id]:
            try:
                await ws.send_json(safe)
            except Exception:
                dead.add(ws)
                self.logger.debug("WS broadcast failed, marking client as dead")
        self._last_broadcast[session_id] = time.time()
        for ws in dead:
            self._connections[session_id].discard(ws)
        if session_id in self._connections and not self._connections[session_id]:
            del self._connections[session_id]

    def is_connected(self, session_id: str) -> bool:
        return session_id in self._connections and bool(self._connections[session_id])


ws_manager = WebSocketManager()
