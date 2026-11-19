import asyncio
import logging
from typing import Any, Dict, Optional, Set
from fastapi import WebSocket

logger = logging.getLogger(__name__)

SEND_TIMEOUT_SECONDS = 5.0


class ConnectionManager:
    """Manages active WebSocket subscriptions per job."""

    def __init__(self):
        self._connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, job_id: str, websocket: WebSocket) -> None:
        """Accept new WebSocket connection and register under job_id."""
        await websocket.accept()
        if job_id not in self._connections:
            self._connections[job_id] = set()
        self._connections[job_id].add(websocket)
        logger.debug(f"WS client connected for job {job_id}. Total: {len(self._connections[job_id])}")

    def disconnect(self, job_id: str, websocket: WebSocket) -> None:
        """Remove WebSocket connection on disconnect."""
        if job_id in self._connections:
            self._connections[job_id].discard(websocket)
            if not self._connections[job_id]:
                del self._connections[job_id]
        logger.debug(f"WS client disconnected for job {job_id}")

    async def broadcast(self, job_id: str, payload: Dict[str, Any]) -> None:
        """Broadcast JSON payload to all active WebSockets connected to this job."""
        subscribers = list(self._connections.get(job_id, ()))
        if not subscribers:
            return

        async def _send(ws: WebSocket) -> Optional[WebSocket]:
            if ws not in self._connections.get(job_id, ()):
                return None
            try:
                # The pipeline awaits this: a client that stopped reading (a suspended window, a
                # half-open connection) must not stall the job, so it is dropped instead.
                await asyncio.wait_for(ws.send_json(payload), timeout=SEND_TIMEOUT_SECONDS)
                return None
            except Exception as e:
                logger.warning(f"Failed to send WS message for job {job_id}: {e!r}")
                return ws

        dead = await asyncio.gather(*(_send(ws) for ws in subscribers))
        for ws in dead:
            if ws is not None:
                self.disconnect(job_id, ws)
                try:
                    await asyncio.wait_for(ws.close(code=1011), timeout=1.0)
                except Exception:
                    pass


ws_manager = ConnectionManager()
