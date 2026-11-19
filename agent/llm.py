"""LLM adapters.

Two implementations share one interface:

* ``OpenAIChatClient`` — any OpenAI-compatible /chat/completions endpoint
  (OpenAI, DeepSeek, vLLM, Ollama's OpenAI shim, ...). Uses urllib so the
  core has no hard dependency beyond the standard library.
* ``MockLLMClient`` — a deterministic scripted planner that lets the demo
  run fully offline and makes the test suite reproducible. It plans the
  same ReAct trajectory a real model would: pick a tool, read the
  observation, pick the next tool, then answer.
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Dict, List, Optional


class LLMClient:
    """Interface: given messages + tool schemas, return one assistant message."""

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAIChatClient(LLMClient):
    """Minimal OpenAI-compatible client (no SDK dependency)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.2,
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages, tools):
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode())
        return body["choices"][0]["message"]


# --------------------------------------------------------------------------
# Mock planner: keyword-driven scripted trajectories, same shape as real
# function-calling responses so the agent loop cannot tell them apart.
# --------------------------------------------------------------------------

_CALC_NUMBERS = re.compile(r"-?\d+(?:\.\d+)?")
_DOC_HINTS = ("doc", "refund", "policy", "price", "plan", "sla", "search", "find", "how")
_CALC_HINTS = ("%", "percent", "calculate", "compute", "how much", "sum", "average", "total")


class MockLLMClient(LLMClient):
    """Scripted ReAct planner for offline demos.

    Trajectory rule:
    1. If the task references the knowledge base -> call ``knowledge_search``.
    2. If the task also needs arithmetic -> afterwards call ``calculator``
       using numbers observed in the search result or the task itself.
    3. Tasks with arithmetic only -> single ``calculator`` call.
    4. Otherwise -> answer directly from the task text.
    """

    def __init__(self) -> None:
        self._docs: Dict[str, str] = {}

    def seed_docs(self, docs: Dict[str, str]) -> None:
        """Give the mock a tiny corpus so knowledge_search observations are real."""
        self._docs = dict(docs)

    def chat(self, messages, tools):
        task = _first_user_task(messages)
        last_obs = _last_observation(messages)
        called = _called_tools(messages)
        available = {t["function"]["name"] for t in tools}

        if last_obs is None:
            # Step 1: decide the first action.
            needs_docs = any(h in task.lower() for h in _DOC_HINTS)
            needs_calc = any(h in task.lower() for h in _CALC_HINTS)
            if needs_docs and "knowledge_search" in available:
                return _tool_call("knowledge_search", {"query": _doc_query(task)})
            if needs_calc and "calculator" in available:
                return _tool_call("calculator", {"expression": _calc_expression(task)})
            return _final(f"Here is the answer regarding '{task}': based on the task description, the mock agent responds directly without tool use.")

        # Step 2+: a tool already ran.
        docs_used = "knowledge_search" in called
        docs_match = any(h in task.lower() for h in _DOC_HINTS)
        calc_hint = any(h in task.lower() for h in _CALC_HINTS)
        if docs_used and docs_match and calc_hint and "calculator" in available and "calculator" not in called:
            expr = _follow_up_expression(task, last_obs)
            if expr:
                return _tool_call("calculator", {"expression": expr})
        return _final(_synthesize(task, called, last_obs))


def _first_user_task(messages: List[Dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") == "user":
            c = m.get("content") or ""
            if isinstance(c, str):
                return c
    return ""


def _last_observation(messages: List[Dict[str, Any]]) -> Optional[str]:
    for m in reversed(messages):
        if m.get("role") == "tool":
            return str(m.get("content", ""))
    return None


def _called_tools(messages: List[Dict[str, Any]]) -> List[str]:
    names = []
    for m in messages:
        for call in m.get("tool_calls") or []:
            fn = call.get("function", {})
            if fn.get("name"):
                names.append(fn["name"])
    return names


def _doc_query(task: str) -> str:
    stop = {"what", "is", "the", "a", "an", "for", "in", "of", "our", "on", "how", "does", "do", "search", "docs", "documentation", "and", "to"}
    words = [w for w in re.findall(r"[a-zA-Z]+", task.lower()) if w not in stop]
    return " ".join(words[:4]) if words else task


def _calc_expression(task: str) -> str:
    nums = _CALC_NUMBERS.findall(task)
    if "%" in task or "percent" in task.lower():
        if len(nums) >= 2:
            return f"{nums[-1]}*{nums[-2]}/100"
    if "average" in task.lower() and len(nums) >= 2:
        return "+".join(nums) + f"/{len(nums)}"
    if len(nums) >= 2 and any(op in task for op in ("+", "-", "*", "sum", "total")):
        return "+".join(nums)
    return nums[0] if nums else "0"


def _follow_up_expression(task: str, observation: str) -> Optional[str]:
    """Derive a follow-up calculation from the task + the doc that was found.

    The mock is scripted, so it encodes the one derivation a real model
    would perform from the pricing FAQ: unit price × user count × months.
    Returns None when the task does not fit a known pattern, in which case
    the mock answers directly instead of fabricating a calculation.
    """
    t = task.lower()
    users = _CALC_NUMBERS.findall(t)
    # pattern: "<plan> plan cost per year for N users"
    if "year" in t and "user" in t and len(users) >= 1:
        n = users[-1]
        price = None
        if "team" in t:
            price = "49"
        elif "starter" in t:
            price = "19"
        elif "enterprise" in t:
            price = "1200"
        if price:
            return f"{price}*{n}*12"
    # pattern: percent of an amount found in the docs
    if "%" in t or "percent" in t:
        nums = _CALC_NUMBERS.findall(t)
        if len(nums) >= 2:
            return f"{nums[-1]}*{nums[-2]}/100"
    return None


def _combine_numbers(nums: List[str], task: str) -> str:
    if "%" in task or "percent" in task.lower():
        if len(nums) >= 2:
            return f"{nums[1]}*{nums[0]}/100"
    if len(nums) >= 2 and "average" in task.lower():
        return "+".join(nums) + f"/{len(nums)}"
    if len(nums) >= 2:
        return "+".join(nums)
    return nums[0] if nums else "0"


def _tool_call(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": f"call_{name}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    }


def _final(text: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": text, "tool_calls": None}


def _synthesize(task: str, called: List[str], last_obs: str) -> str:
    parts = [f"Completed task: {task.rstrip('.')}."]
    if "knowledge_search" in called:
        parts.append("The knowledge base matched a relevant document.")
    if "calculator" in called:
        parts.append(f"Computed result: {last_obs}")
    if not called:
        parts.append("No tools were required; answered directly.")
    return " ".join(parts)
