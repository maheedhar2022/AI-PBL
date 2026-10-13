"""
WebSocket route for live monitoring events.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter(tags=["websocket"])

# Connected WebSocket clients
_clients: Set[WebSocket] = set()


async def broadcast_ws(message: dict) -> None:
    """Broadcast a message to all connected WebSocket clients."""
    if not _clients:
        return
    payload = json.dumps(message)
    dead: Set[WebSocket] = set()
    for ws in _clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)


@router.websocket("/ws/live")
async def websocket_live(websocket: WebSocket) -> None:
    await websocket.accept()
    _clients.add(websocket)
    logger.info(f"WebSocket client connected. Total: {len(_clients)}")
    try:
        while True:
            # Keep connection alive; receive pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    finally:
        _clients.discard(websocket)

