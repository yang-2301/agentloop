"""Conversation memory with a bounded window.

Keeps the system prompt and the original task always visible while
trimming the oldest (thought, action, observation) turns once the
history grows past ``max_messages``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Memory:
    """Ordered message list for the ReAct conversation."""

    system: str
    max_messages: int = 40
    _messages: List[Dict[str, Any]] = field(default_factory=list)

    def append(self, message: Dict[str, Any]) -> None:
        self._messages.append(message)
        self._trim()

    def extend(self, messages: List[Dict[str, Any]]) -> None:
        for m in messages:
            self.append(m)

    def _trim(self) -> None:
        # +1 because every ReAct turn adds an assistant message and a
        # tool message; trim in pairs so tool results stay attached.
        overflow = len(self._messages) - self.max_messages
        if overflow > 0:
            drop = overflow + (overflow % 2)
            self._messages = self._messages[drop:]

    def render(self) -> List[Dict[str, Any]]:
        return [{"role": "system", "content": self.system}] + list(self._messages)
