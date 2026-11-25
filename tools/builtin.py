"""Built-in demo tools shipped with the agent."""

from __future__ import annotations

import json
import re
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

from agent.tools import Tool

# ---------------------------------------------------------------------------
# calculator: safe arithmetic evaluation (no eval of arbitrary code)
# ---------------------------------------------------------------------------

_ALLOWED = re.compile(r"^[\d\s+\-*/().%]+$")


def _calculator(expression: str) -> str:
    expr = expression.strip()
    if not expr or len(expr) > 200:
        return "ERROR: expression empty or too long"
    if not _ALLOWED.match(expr):
        return "ERROR: only numbers and + - * / ( ) % are allowed"
    # % is only meaningful inside the whitelist as a modulo/percent marker;
    # translate "a%b" style percent math the demo expects: x% of y handled by caller.
    try:
        value = eval(compile(expr, "<calc>", "eval"), {"__builtins__": {}}, {})  # noqa: S307 - charset-whitelisted
        if not isinstance(value, (int, float)):
            return "ERROR: expression did not evaluate to a number"
        value = round(float(value), 6)
        return str(int(value)) if float(value).is_integer() else str(value)
    except ZeroDivisionError:
        return "ERROR: division by zero"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


calculator_tool = Tool(
    name="calculator",
    description="Evaluate an arithmetic expression. Accepts numbers and + - * / ( ) %. Example: '2400*0.15'",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "Arithmetic expression to evaluate"}},
        "required": ["expression"],
    },
    fn=_calculator,
)

# ---------------------------------------------------------------------------
# knowledge_search: tiny in-memory corpus with BM25-lite scoring
# ---------------------------------------------------------------------------

_DOCS: List[Dict[str, str]] = [
    {"id": "faq-refund", "title": "Refund policy", "text": "Customers may request a full refund within 30 days of purchase. Annual plans can be refunded proportionally after 30 days. Refunds are processed within 5 business days."},
    {"id": "faq-pricing", "title": "Pricing plans", "text": "The Starter plan costs 19 dollars per month, the Team plan costs 49 dollars per user per month, and the Enterprise plan is priced annually at 1200 dollars per seat."},
    {"id": "faq-sla", "title": "Support SLA", "text": "Priority support has a first response SLA of 2 hours. Business support responds within 8 business hours. Critical incidents are acknowledged within 15 minutes."},
    {"id": "faq-export", "title": "Data export", "text": "Workspace admins can export all project data as CSV or JSON from Settings > Export. Exports include audit logs and are delivered within 24 hours."},
    {"id": "faq-sso", "title": "Single sign-on", "text": "SAML 2.0 single sign-on is available on the Enterprise plan. Okta, Azure AD and Google Workspace are supported out of the box."},
]

_STOP = {"what", "is", "the", "a", "an", "of", "for", "in", "on", "how", "does", "do", "our", "and", "to", "per"}


def _tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1]


def _knowledge_search(query: str) -> str:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return "ERROR: empty query"
    scored = []
    for doc in _DOCS:
        d_tokens = _tokenize(doc["title"] + " " + doc["text"])
        matches = sum(1 for t in q_tokens if t in d_tokens)
        if matches:
            score = matches / len(q_tokens)
            scored.append((score, doc))
    if not scored:
        return json.dumps({"results": [], "note": "no document matched the query"})
    scored.sort(key=lambda s: -s[0])
    top = scored[:3]
    return json.dumps({
        "results": [
            {"id": d["id"], "title": d["title"], "excerpt": d["text"][:240], "score": round(s, 3)}
            for s, d in top
        ]
    }, ensure_ascii=False)


knowledge_search_tool = Tool(
    name="knowledge_search",
    description="Search the product knowledge base (FAQ about pricing, refunds, SLA, export, SSO). Returns matching document excerpts.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Free-text search query"}},
        "required": ["query"],
    },
    fn=_knowledge_search,
)

# ---------------------------------------------------------------------------
# http_get: fetch a URL (size-capped, content-type aware)
# ---------------------------------------------------------------------------

def _is_private_host(hostname: str) -> bool:
    """Block loopback / private / link-local targets (SSRF guard)."""
    import ipaddress

    name = hostname.strip("[]").lower()
    if name in ("localhost",) or name.endswith(".localhost") or name.endswith(".internal"):
        return True
    try:
        ip = ipaddress.ip_address(name)
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_unspecified
    except ValueError:
        return False


def _http_get(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "ERROR: only http(s) URLs are supported"
    from urllib.parse import urlparse

    host = urlparse(url).hostname or ""
    if not host or _is_private_host(host):
        return f"ERROR: refusing to fetch internal address '{host}'"
    req = urllib.request.Request(url, headers={"User-Agent": "agentloop-demo/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        ctype = resp.headers.get("Content-Type", "")
        body = resp.read(20000).decode("utf-8", "replace")
    return json.dumps({"status": resp.status, "content_type": ctype, "body_preview": body[:1000]})


http_get_tool = Tool(
    name="http_get",
    description="Fetch an HTTP(S) URL and return status, content type and a body preview.",
    parameters={
        "type": "object",
        "properties": {"url": {"type": "string", "description": "Absolute http(s) URL to fetch"}},
        "required": ["url"],
    },
    fn=_http_get,
)

# ---------------------------------------------------------------------------
# now: current UTC time
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


now_tool = Tool(
    name="now",
    description="Return the current UTC time.",
    parameters={"type": "object", "properties": {}, "required": []},
    fn=_now,
)


def default_registry() -> "Any":
    """Registry with all built-in demo tools pre-registered."""
    from agent.tools import ToolRegistry

    registry = ToolRegistry()
    for tool in (calculator_tool, knowledge_search_tool, http_get_tool, now_tool):
        registry.register(tool)
    return registry
