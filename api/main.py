"""FastAPI service: run the agent over HTTP, list tools, inspect traces."""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import MockLLMClient, OpenAIChatClient, ReActAgent
from tools.builtin import default_registry

app = FastAPI(title="agentloop", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_runs: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()


def build_agent() -> ReActAgent:
    """Mock mode unless OPENAI_API_KEY is set (documented in README)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        llm = OpenAIChatClient(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=api_key,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
    else:
        llm = MockLLMClient()
    return ReActAgent(llm=llm, tools=default_registry())


class RunRequest(BaseModel):
    task: str = Field(min_length=1, max_length=2000)
    max_steps: int = Field(default=8, ge=1, le=20)


@app.post("/api/run")
def run_agent(req: RunRequest) -> Dict[str, Any]:
    agent = build_agent()
    result = agent.run(req.task)
    with _lock:
        _runs[result.run_id] = result.as_dict()
    return result.as_dict()


@app.get("/api/runs")
def list_runs() -> List[Dict[str, Any]]:
    with _lock:
        runs = sorted(_runs.values(), key=lambda r: -r["duration_ms"])
        return [
            {"run_id": r["run_id"], "task": r["task"], "status": r["status"],
             "tool_calls_total": r["tool_calls_total"], "duration_ms": r["duration_ms"]}
            for r in runs
        ]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    with _lock:
        run = _runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@app.get("/api/tools")
def list_tools() -> Dict[str, Any]:
    registry = default_registry()
    return {"tools": [{"name": n, "description": t.description} for n, t in
                      zip(registry.names(), [registry.get(n) for n in registry.names()])]}


@app.get("/api/health")
def health() -> Dict[str, str]:
    mode = "openai" if os.environ.get("OPENAI_API_KEY") else "mock"
    return {"status": "ok", "llm_mode": mode}
