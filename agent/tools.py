"""Tool registry for the agent loop.

Tools are plain Python callables wrapped in a schema the LLM can see.
The schema mirrors the OpenAI function-calling format so the same
registry works against OpenAI-compatible providers and the offline mock.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Tool:
    """A callable the agent can decide to invoke.

    ``parameters`` is a JSON-Schema dict describing the arguments object.
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    fn: Callable[..., Any]

    def openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, raw_args: str) -> str:
        """Execute the tool with a JSON arguments string, return a string result.

        Malformed arguments never raise out of the loop: they become an
        error observation the model can read and recover from.
        """
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError as exc:
            return f"ERROR: arguments are not valid JSON: {exc}"
        if not isinstance(args, dict):
            return "ERROR: arguments must be a JSON object"
        missing = [
            k for k, v in self.parameters.get("properties", {}).items()
            if k in self.parameters.get("required", []) and k not in args
        ]
        if missing:
            return f"ERROR: missing required argument(s): {', '.join(missing)}"
        try:
            result = self.fn(**args)
        except Exception as exc:  # noqa: BLE001 - observations must stay strings
            return f"ERROR: {type(exc).__name__}: {exc}"
        return _stringify(result)


def _stringify(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False)
    except TypeError:
        return str(result)


@dataclass
class ToolRegistry:
    """Named collection of tools, keyed by tool name."""

    _tools: Dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.openai_schema() for t in self._tools.values()]
