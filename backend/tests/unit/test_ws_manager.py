"""A WebSocket client that stopped reading must not stall the pipeline that broadcasts to it."""
import asyncio
import time

import pytest

import app.core.ws_manager as wsm


class FakeSocket:
    def __init__(self, hang=False):
        self.hang = hang
        self.sent = []
        self.closed = False

    async def accept(self):
        pass

    async def send_json(self, payload):
        if self.hang:
            await asyncio.sleep(3600)
        self.sent.append(payload)

    async def close(self, code=1000):
        self.closed = True


@pytest.mark.asyncio
async def test_a_stuck_client_is_dropped_and_others_still_get_the_message(monkeypatch):
    monkeypatch.setattr(wsm, "SEND_TIMEOUT_SECONDS", 0.2)
    manager = wsm.ConnectionManager()
    stuck, healthy = FakeSocket(hang=True), FakeSocket()
    await manager.connect("job", stuck)
    await manager.connect("job", healthy)

    began = time.monotonic()
    await manager.broadcast("job", {"event": "STAGE_PROGRESS"})
    assert time.monotonic() - began < 2
    assert healthy.sent == [{"event": "STAGE_PROGRESS"}]
    assert stuck.closed

    await manager.broadcast("job", {"event": "JOB_COMPLETED"})  # the stuck one is gone
    assert healthy.sent[-1] == {"event": "JOB_COMPLETED"}
