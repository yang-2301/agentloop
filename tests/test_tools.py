"""Unit tests for the tool registry and built-in tools."""

import json

import pytest

from agent import Tool, ToolRegistry
from tools.builtin import calculator_tool, knowledge_search_tool, now_tool


def echo_tool():
    return Tool(
        name="echo",
        description="echo text",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        fn=lambda text: text,
    )


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        reg.register(echo_tool())
        assert reg.names() == ["echo"]
        assert reg.get("echo").name == "echo"
        assert reg.get("nope") is None

    def test_duplicate_registration_rejected(self):
        reg = ToolRegistry()
        reg.register(echo_tool())
        with pytest.raises(ValueError):
            reg.register(echo_tool())

    def test_openai_schema_shape(self):
        schema = echo_tool().openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "echo"
        assert "properties" in schema["function"]["parameters"]

    def test_run_valid_json(self):
        assert echo_tool().run(json.dumps({"text": "hi"})) == "hi"

    def test_run_invalid_json_is_error_observation(self):
        out = echo_tool().run("{not json")
        assert out.startswith("ERROR:")

    def test_run_missing_required_argument(self):
        out = echo_tool().run("{}")
        assert "missing required argument" in out

    def test_run_non_json_object(self):
        out = echo_tool().run("[1,2]")
        assert out.startswith("ERROR:")

    def test_tool_exception_becomes_observation(self):
        def boom():
            raise RuntimeError("nope")
        tool = Tool(name="boom", description="", parameters={"type": "object", "properties": {}}, fn=boom)
        assert tool.run("{}").startswith("ERROR: RuntimeError")


class TestCalculator:
    def test_basic_arithmetic(self):
        assert calculator_tool.run('{"expression": "2+3*4"}') == "14"

    def test_percent_expression(self):
        assert calculator_tool.run('{"expression": "2400*0.15"}') == "360"

    def test_rejects_letters(self):
        out = calculator_tool.run('{"expression": "import os"}')
        assert out.startswith("ERROR:")

    def test_division_by_zero(self):
        assert "division by zero" in calculator_tool.run('{"expression": "1/0"}')

    def test_empty_expression(self):
        assert calculator_tool.run('{"expression": ""}').startswith("ERROR:")


class TestKnowledgeSearch:
    def test_finds_refund_doc(self):
        out = json.loads(knowledge_search_tool.run('{"query": "refund policy"}'))
        assert out["results"], "should return at least one result"
        assert any(r["id"] == "faq-refund" for r in out["results"])

    def test_finds_pricing_doc(self):
        out = json.loads(knowledge_search_tool.run('{"query": "how much does the team plan cost"}'))
        assert any(r["id"] == "faq-pricing" for r in out["results"])

    def test_no_match_returns_empty(self):
        out = json.loads(knowledge_search_tool.run('{"query": "zzzqqq"}'))
        assert out["results"] == []

    def test_empty_query_errors(self):
        assert knowledge_search_tool.run('{"query": ""}').startswith("ERROR:")


class TestNow:
    def test_returns_string(self):
        assert "UTC" in now_tool.run("{}")


class TestHttpGetSSRFGuard:
    """The fetch tool must refuse internal targets (offline checks only)."""

    def test_blocks_localhost(self):
        from tools.builtin import http_get_tool
        out = http_get_tool.run('{"url": "http://localhost:8124/api/health"}')
        assert out.startswith("ERROR:")
        assert "internal address" in out

    def test_blocks_private_ip(self):
        from tools.builtin import http_get_tool
        out = http_get_tool.run('{"url": "http://192.168.1.1/admin"}')
        assert "internal address" in out

    def test_blocks_metadata_service(self):
        from tools.builtin import http_get_tool
        out = http_get_tool.run('{"url": "http://169.254.169.254/latest/meta-data/"}')
        assert "internal address" in out

    def test_blocks_non_http_scheme(self):
        from tools.builtin import http_get_tool
        assert http_get_tool.run('{"url": "ftp://example.com/x"}').startswith("ERROR:")
