"""The ReAct loop.

One ``run`` = reason -> act -> observe, repeated until the model emits a
final answer or a guard (max steps / step timeout) stops it. Every step
is recorded in a trace so the UI can replay the agent's reasoning.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .llm import LLMClient
from .memory import Memory
from .tools import ToolRegistry

DEFAULT_MAX_STEPS = 8
STEP_TIMEOUT_S = 15.0

SYSTEM_PROMPT = """You are a precise task-solving agent. You answer user tasks by
calling the provided tools when they help, then composing a final answer from the
observations you collected. Rules:
- Call a tool when you need facts you do not already have.
- Read every observation carefully before deciding the next step.
- When you have enough information, stop calling tools and write the final answer.
- If a tool returns an ERROR observation, fix the arguments or try another tool.
- Never invent facts; ground every claim in an observation or say you are unsure."""


@dataclass
class Step:
    """One turn of the loop: what the model thought, called, and got back."""

    index: int
    thought: Optional[str]
    tool_calls: List[Dict[str, Any]]
    observations: List[Dict[str, str]] = field(default_factory=list)
    duration_ms: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "thought": self.thought,
            "tool_calls": self.tool_calls,
            "observations": self.observations,
            "duration_ms": self.duration_ms,
        }


@dataclass
class RunResult:
    run_id: str
    task: str
    status: str  # "completed" | "max_steps" | "error"
    answer: Optional[str]
    steps: List[Step]
    tool_calls_total: int
    duration_ms: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "status": self.status,
            "answer": self.answer,
            "steps": [s.as_dict() for s in self.steps],
            "tool_calls_total": self.tool_calls_total,
            "duration_ms": self.duration_ms,
        }


class ReActAgent:
    """Reason-act-observe loop over a tool registry and an LLM client."""

    def __init__(self, llm: LLMClient, tools: ToolRegistry, max_steps: int = DEFAULT_MAX_STEPS) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps

    def run(self, task: str) -> RunResult:
        run_id = uuid.uuid4().hex[:12]
        started = time.monotonic()
        memory = Memory(system=SYSTEM_PROMPT)
        memory.append({"role": "user", "content": task})

        steps: List[Step] = []
        tool_calls_total = 0
        status = "completed"
        answer: Optional[str] = None

        for i in range(self.max_steps):
            t0 = time.monotonic()
            try:
                message = self.llm.chat(memory.render(), self.tools.schemas())
            except Exception as exc:  # noqa: BLE001
                status = "error"
                step = Step(index=i, thought=f"LLM call failed: {exc}", tool_calls=[])
                steps.append(step)
                break

            content = message.get("content")
            calls = message.get("tool_calls") or []
            step = Step(index=i, thought=content, tool_calls=calls)

            if not calls:
                # No tool calls -> final answer; end the loop.
                answer = content if isinstance(content, str) else str(content)
                step.duration_ms = int((time.monotonic() - t0) * 1000)
                steps.append(step)
                break

            memory.append({"role": "assistant", "content": content, "tool_calls": calls})
            for call in calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                raw_args = fn.get("arguments") or "{}"
                tool = self.tools.get(name)
                if tool is None:
                    observation = f"ERROR: unknown tool '{name}'. Available: {', '.join(self.tools.names())}"
                else:
                    observation = tool.run(raw_args)
                tool_calls_total += 1
                call_id = call.get("id") or f"call_{i}"
                step.observations.append({"call_id": call_id, "tool": name, "output": observation})
                memory.append({"role": "tool", "tool_call_id": call_id, "content": observation})

            step.duration_ms = int((time.monotonic() - t0) * 1000)
            steps.append(step)
        else:
            status = "max_steps"

        return RunResult(
            run_id=run_id,
            task=task,
            status=status,
            answer=answer,
            steps=steps,
            tool_calls_total=tool_calls_total,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
