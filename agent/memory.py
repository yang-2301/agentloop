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
        overflow = len(self._messages) - self.max_messages
        if overflow <= 0:
            return
        # The first user message is the original task: keep it visible for
        # the whole run. Everything after it is (assistant, tool) pairs —
        # trim whole pairs only, so observations stay attached to the
        # assistant message whose tool_calls produced them.
        has_task = bool(self._messages) and self._messages[0].get("role") == "user"
        head = self._messages[:1] if has_task else []
        body = self._messages[1:] if has_task else self._messages
        drop = min(overflow + (overflow % 2), len(body) - (len(body) % 2))
        self._messages = head + body[drop:]

    def render(self) -> List[Dict[str, Any]]:
        return [{"role": "system", "content": self.system}] + list(self._messages)
