"""OpenClaw-style blackboard for autonomous FactoryMind agents."""

from __future__ import annotations

import time


class Blackboard:
    """Small in-process message board shared by the leader and worker loop."""

    def __init__(self) -> None:
        self._messages: list[dict] = []

    def post(self, from_id: int, msg_type: str, content: str, **extra) -> None:
        message = {
            "from": from_id,
            "type": msg_type,
            "content": content,
            "timestamp": time.time(),
        }
        message.update(extra)
        self._messages.append(message)

    def read_recent(self, n: int = 20) -> list[dict]:
        return self._messages[-n:]

    def read_by_type(self, msg_type: str) -> list[dict]:
        return [message for message in self._messages if message["type"] == msg_type]

    def read_recent_by_type(self, msg_type: str, seconds: float = 30.0) -> list[dict]:
        cutoff = time.time() - seconds
        return [
            message for message in self._messages
            if message["type"] == msg_type and message.get("timestamp", 0.0) >= cutoff
        ]

    def clear(self) -> None:
        self._messages.clear()

    def to_list(self) -> list[dict]:
        return list(self._messages)
