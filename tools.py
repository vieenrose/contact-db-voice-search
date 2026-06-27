#!/usr/bin/env python3
"""LiveKit-style function-tool calling — without LiveKit.

Mirrors the ergonomics of LiveKit Agents' `@function_tool`: decorate a plain
Python function, and its signature + docstring become an OpenAI-compatible JSON
tool schema. A small agentic loop then parses the model's tool-calls, invokes the
function, and feeds the result back — until the model produces a final reply.

The point is identical tool *semantics* to LiveKit (same registry, same OpenAI
tool-call shape) with zero LiveKit dependency, so the demo runs anywhere — plain
HF Space, CPU, or keystone. Porting to real LiveKit later is then a transport
swap, not a rewrite: register the same functions with `livekit.agents.function_tool`.

The parser accepts BOTH:
  * Hermes text   <tool_call>{"name":..,"arguments":{..}}</tool_call>   (what our
    fine-tuned Qwen-Omni emits), and
  * OpenAI JSON   message["tool_calls"]                                 (what
    llama-server / the OpenAI API return) — the same bridge LiveKit uses.

Run `python tools.py` for a self-test (no model / no GPU required).
"""
import inspect
import json
import re
from pathlib import Path
from typing import get_type_hints

ROOT = Path(__file__).resolve().parent

_PY2JSON = {str: "string", int: "integer", float: "number",
            bool: "boolean", dict: "object", list: "array"}


class Tool:
    """A registered function plus its auto-derived OpenAI tool schema."""

    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__
        sig = inspect.signature(fn)
        hints = get_type_hints(fn)
        doc = (fn.__doc__ or "").strip()
        props, required = {}, []
        for p in sig.parameters.values():
            if p.name == "self":
                continue
            props[p.name] = {"type": _PY2JSON.get(hints.get(p.name, str), "string")}
            if p.default is inspect.Parameter.empty:
                required.append(p.name)
        self.schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": doc.split("\n")[0],
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        }

    def __call__(self, **kwargs):
        return self.fn(**kwargs)


class ToolRegistry:
    """Holds tools; exposes OpenAI-style schemas and dispatches calls by name."""

    def __init__(self):
        self._tools = {}

    def tool(self, fn):
        """Decorator: register `fn` as a tool, leaving it directly callable too."""
        t = Tool(fn)
        self._tools[t.name] = t
        return fn

    def schemas(self):
        """OpenAI-compatible `tools=[...]` list (also what LiveKit builds internally)."""
        return [t.schema for t in self._tools.values()]

    def dispatch(self, name, arguments=None):
        if name not in self._tools:
            return {"error": f"unknown tool {name!r}"}
        try:
            return self._tools[name](**(arguments or {}))
        except TypeError as e:
            return {"error": f"bad arguments for {name!r}: {e}"}


# --- tool-call parsing: accept both our Hermes text form and OpenAI JSON ----
_HERMES = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_tool_calls(output):
    """Extract [{name, arguments}, ...] from a model output.

    `output` may be a str (Hermes `<tool_call>` text) or an OpenAI-style
    assistant message dict carrying `tool_calls`.
    """
    calls = []
    if isinstance(output, dict):                       # OpenAI message
        for tc in output.get("tool_calls") or []:
            fn = tc.get("function", tc)
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append({"name": fn.get("name"), "arguments": args})
        return calls
    for m in _HERMES.finditer(output or ""):           # Hermes text
        try:
            d = json.loads(m.group(1))
            calls.append({"name": d.get("name"), "arguments": d.get("arguments", {})})
        except json.JSONDecodeError:
            pass
    return calls


def run_tool_loop(generate, registry, max_hops=3):
    """Drive the agentic loop with a model-agnostic `generate` callback.

    `generate(tool_messages)` is called with the running list of
    {role: "tool", name, content} results so far and must return the model's
    next raw output (str or OpenAI message). The loop dispatches any tool-calls,
    appends their results, and re-generates until the model replies with no
    further tool-call (or `max_hops` is hit). Returns (final_text, transcript).
    """
    tool_msgs, last_text = [], ""
    for _ in range(max_hops + 1):
        out = generate(tool_msgs)
        calls = parse_tool_calls(out)
        last_text = out if isinstance(out, str) else (out.get("content") or "")
        if not calls:
            return last_text, tool_msgs
        for c in calls:
            result = registry.dispatch(c["name"], c["arguments"])
            tool_msgs.append({"role": "tool", "name": c["name"],
                              "content": json.dumps(result, ensure_ascii=False)})
    return last_text, tool_msgs


# ===========================================================================
# The attendant's tools — the directory lookup that powers the demo, plus the
# call-control stubs a LiveKit SIP attendant would register (transfer / hangup).
# ===========================================================================
registry = ToolRegistry()
_RESOLVER = None


def _dir_csv():
    """Locate directory.csv whether we run from the repo (data/…) or the Space (./…)."""
    for p in (ROOT / "directory.csv", ROOT / "data" / "directory.csv"):
        if p.exists():
            return str(p)
    return str(ROOT / "data" / "directory.csv")


def _resolver():
    global _RESOLVER
    if _RESOLVER is None:
        from resolver import Resolver
        _RESOLVER = Resolver(_dir_csv())
    return _RESOLVER


def format_matches(res):
    """Resolver result -> the ranked-match list handed back to the model as the tool
    response. This is the EXACT shape the agent was trained on (see generate_dialogs):
    one entry for a confident hit, the close candidates to clarify, or [] for a miss.
    Keeping this identical to training is what lets our fine-tuned model read it."""
    if res["action"] == "resolve":
        return [{"name": res["name"], "ext": res["ext"],
                 "dept": res.get("dept"), "score": res.get("score")}]
    return res.get("candidates", [])          # clarify -> candidates ; not_found -> []


@registry.tool
def search_contacts(query: str, department: str = "") -> list:
    """Look up a colleague by spoken name, optionally narrowed by department."""
    filters = {"department": department} if department else None
    return format_matches(_resolver().resolve(query, filters=filters))


@registry.tool
def transfer_call(extension: str) -> dict:
    """Connect the caller to the given extension (stub; wired to the PBX in production)."""
    return {"status": "transferring", "extension": extension}


@registry.tool
def end_call(reason: str = "completed") -> dict:
    """Hang up the call."""
    return {"status": "ended", "reason": reason}


if __name__ == "__main__":
    print("=== OpenAI-compatible tool schemas (what we'd hand the LLM / LiveKit) ===")
    print(json.dumps(registry.schemas(), indent=2, ensure_ascii=False))

    # 1) Hermes text form — exactly what our fine-tuned Qwen-Omni emits.
    sample = '<tool_call>{"name":"search_contacts","arguments":{"query":"陳凱文"}}</tool_call>'
    print("\n=== parse + dispatch (Hermes text) ===")
    for c in parse_tool_calls(sample):
        print(" call:", c, "->", registry.dispatch(c["name"], c["arguments"]))

    # 2) OpenAI tool_calls form — what llama-server returns (the LiveKit bridge).
    oai = {"role": "assistant", "tool_calls": [
        {"function": {"name": "search_contacts", "arguments": '{"query": "Kevin"}'}}]}
    print("\n=== parse + dispatch (OpenAI tool_calls) ===")
    for c in parse_tool_calls(oai):
        print(" call:", c, "->", registry.dispatch(c["name"], c["arguments"]))

    # 3) The agentic loop end-to-end with a scripted 2-hop model.
    script = iter([
        '<tool_call>{"name":"search_contacts","arguments":{"query":"Kevin Chen"}}</tool_call>',
        "Connecting you now.",
    ])
    print("\n=== run_tool_loop (scripted model) ===")
    final, transcript = run_tool_loop(lambda msgs: next(script), registry)
    print(" tool results:", transcript)
    print(" final reply :", final)
