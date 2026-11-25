"""Tests for conversation memory and the ReAct loop with scripted LLMs."""

from typing import Any, Dict, List

from agent import Memory, MockLLMClient, ReActAgent, Tool, ToolRegistry


class TestMemory:
    def test_render_starts_with_system(self):
        mem = Memory(system="sys")
        mem.append({"role": "user", "content": "hi"})
        rendered = mem.render()
        assert rendered[0]["role"] == "system"
        assert rendered[0]["content"] == "sys"
        assert rendered[1] == {"role": "user", "content": "hi"}

    def test_trim_keeps_pairs(self):
        mem = Memory(system="s", max_messages=4)
        for i in range(6):
            mem.append({"role": "assistant", "content": f"a{i}"})
            mem.append({"role": "tool", "content": f"o{i}"})
        assert len(mem._messages) == 4
        # trimming keeps newest content: "a5/o5" and "a4/o4"
        contents = [m.get("content") for m in mem._messages]
        assert contents == ["a4", "o4", "a5", "o5"]

    def test_trim_preserves_original_task(self):
        mem = Memory(system="s", max_messages=4)
        mem.append({"role": "user", "content": "the original task"})
        for i in range(4):
            mem.append({"role": "assistant", "content": f"a{i}"})
            mem.append({"role": "tool", "content": f"o{i}"})
        assert len(mem._messages) == 3
        contents = [m.get("content") for m in mem._messages]
        # task stays first; oldest pairs dropped whole, newest pair intact
        assert contents == ["the original task", "a3", "o3"]


class ScriptedLLM:
    """Returns queued responses in order; asserts what it was asked."""

    def __init__(self, responses: List[Dict[str, Any]]):
        self.responses = list(responses)
        self.requests: List[List[Dict[str, Any]]] = []

    def chat(self, messages, tools):
        self.requests.append(messages)
        return self.responses.pop(0)


def tool_call(name: str, args: Dict[str, Any], call_id: str = "c1") -> Dict[str, Any]:
    import json
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}


def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool(name="add", description="", parameters={
        "type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"]}, fn=lambda a, b: a + b))
    return reg


class TestReActLoop:
    def test_tool_then_final_answer(self):
        llm = ScriptedLLM([
            {"role": "assistant", "content": "I should add the numbers.", "tool_calls":
             [{"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'}}]},
            {"role": "assistant", "content": "The sum is 5.", "tool_calls": None},
        ])
        agent = ReActAgent(llm=llm, tools=registry())
        result = agent.run("add 2 and 3")

        assert result.status == "completed"
        assert result.answer == "The sum is 5."
        assert result.tool_calls_total == 1
        assert len(result.steps) == 2
        assert result.steps[0].thought == "I should add the numbers."
        assert result.steps[0].observations[0]["output"] == "5"
        assert result.steps[1].observations == []

    def test_observation_is_fed_back_to_llm(self):
        llm = ScriptedLLM([
            tool_call("add", {"a": 2, "b": 3}),
            {"role": "assistant", "content": "done", "tool_calls": None},
        ])
        agent = ReActAgent(llm=llm, tools=registry())
        agent.run("add 2 and 3")

        second_request = llm.requests[1]
        tool_msg = [m for m in second_request if m.get("role") == "tool"]
        assert len(tool_msg) == 1
        assert tool_msg[0]["content"] == "5"

    def test_unknown_tool_returns_error_observation(self):
        llm = ScriptedLLM([
            tool_call("missing_tool", {}),
            {"role": "assistant", "content": "ok, answered anyway", "tool_calls": None},
        ])
        agent = ReActAgent(llm=llm, tools=registry())
        result = agent.run("do the thing")

        assert result.status == "completed"
        out = result.steps[0].observations[0]["output"]
        assert "unknown tool" in out
        assert "add" in out  # lists available tools

    def test_max_steps_guard(self):
        agent = ReActAgent(llm=ScriptedNeverEnding(tool_call("add", {"a": 1, "b": 1})), tools=registry(), max_steps=3)
        result = agent.run("loop forever")

        assert result.status == "max_steps"
        assert result.answer is None
        assert len(result.steps) == 3
        assert result.tool_calls_total == 3

    def test_llm_failure_marks_error_status(self):
        class BoomLLM:
            def chat(self, messages, tools):
                raise ConnectionError("provider down")

        agent = ReActAgent(llm=BoomLLM(), tools=registry())
        result = agent.run("anything")
        assert result.status == "error"
        assert "provider down" in result.steps[0].thought

    def test_run_result_shape(self):
        llm = ScriptedLLM([{"role": "assistant", "content": "42", "tool_calls": None}])
        result = ReActAgent(llm=llm, tools=registry()).run("meaning of life")
        d = result.as_dict()
        for key in ("run_id", "task", "status", "answer", "steps", "tool_calls_total", "duration_ms"):
            assert key in d


class ScriptedNeverEnding:
    """Always returns the same tool-call response (queue never exhausts)."""

    def __init__(self, response: Dict[str, Any]):
        self.response = response
        self.calls = 0

    def chat(self, messages, tools):
        self.calls += 1
        return self.response


class TestMockLLM:
    def test_mock_runs_multi_step_react_offline(self):
        agent = ReActAgent(llm=MockLLMClient(), tools=_demo_registry())
        result = agent.run("Search our docs: how much does the team plan cost per year for 10 users?")

        assert result.status == "completed"
        assert result.answer
        assert result.tool_calls_total >= 1

    def test_mock_answers_directly_when_no_tool_needed(self):
        agent = ReActAgent(llm=MockLLMClient(), tools=_demo_registry())
        result = agent.run("Introduce yourself in one sentence.")
        assert result.status == "completed"
        assert result.tool_calls_total == 0

    def test_mock_calculator_task(self):
        agent = ReActAgent(llm=MockLLMClient(), tools=_demo_registry())
        result = agent.run("Calculate 15% of 2400")
        assert result.status == "completed"
        assert result.tool_calls_total == 1
        assert result.answer and "360" in result.answer

    def test_mock_error_recovery_trajectory(self):
        agent = ReActAgent(llm=MockLLMClient(), tools=_demo_registry())
        result = agent.run("Search our docs for the refund policy, and demo error recovery")

        assert result.status == "completed"
        assert result.tool_calls_total == 2
        # step 1: unregistered tool -> ERROR observation, not a crash
        err = result.steps[0].observations[0]["output"]
        assert "unknown tool" in err and "web_search" in err
        # step 2: the model reads the error and self-corrects
        assert result.steps[1].tool_calls[0]["function"]["name"] == "knowledge_search"
        assert "recover" in (result.steps[1].thought or "").lower() or "falling back" in (result.steps[1].thought or "")
        assert result.answer


def _demo_registry() -> ToolRegistry:
    from tools.builtin import calculator_tool, knowledge_search_tool, now_tool
    reg = ToolRegistry()
    for t in (calculator_tool, knowledge_search_tool, now_tool):
        reg.register(t)
    return reg
