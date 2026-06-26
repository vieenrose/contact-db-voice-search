---
title: Taiwan Attendant — LiveKit-style Tool Calling
emoji: ☎️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
short_description: Zh-TW/en attendant; LiveKit-style tool calling, no LiveKit
---

# ☎️ Taiwan Office Attendant — LiveKit-style tool calling *(no LiveKit)*

A heard name → the model emits a **`search_contacts` tool call** → the registry **dispatches**
it against the **live contact directory** → the ranked matches come back as the **tool response**
→ the model **connects**, **asks to clarify**, or **rejects** an unknown name (no misroute).

The directory search is the component-aware resolver from the project benchmark
(**92.8% task success / 1.8% misroute, Mandarin 98.4%**). The directory lives in
**`directory.csv`** — *not in the model weights*. Edit a contact and the search updates
instantly, **no retraining**.

## LiveKit-style tool calling — without LiveKit

"LiveKit-style tool calling" is just three small pieces, none of which require LiveKit. They
live in **`tools.py`** and are the *same* code the fine-tuned Qwen-Omni agent uses:

**1. A registry that auto-derives the tool schema** — decorate a plain Python function and its
signature + docstring become an OpenAI-compatible tool schema (exactly what LiveKit builds for
`@function_tool`, and what `llama-server` / the OpenAI API expect):

```python
@registry.tool
def search_contacts(query: str) -> list:
    "Look up a colleague by spoken name and return ranked directory matches."
    return format_matches(_resolver().resolve(query))
```

→ produces `{"type":"function","function":{"name":"search_contacts","parameters":{…}}}`
automatically. (Open the **🔧 Registered tools** panel in the demo to see the live schema —
`search_contacts` plus `transfer_call` / `end_call`, the call-control tools a SIP attendant adds.)

**2. A parser that reads both tool-call formats:**
- **Hermes text** `<tool_call>{…}</tool_call>` — what our fine-tuned Qwen-Omni emits
- **OpenAI `tool_calls`** JSON — what `llama-server` / the OpenAI API return

So one layer drives our model today and an OpenAI-compatible endpoint tomorrow — the same bridge
LiveKit itself uses.

**3. An agentic loop** (`run_tool_loop`): generate → parse the tool-call → dispatch via the
registry → feed the tool response back → repeat until the model gives a plain reply.

## How this maps to LiveKit

| LiveKit Agents | Here (`tools.py`) |
|---|---|
| `@function_tool` | `@registry.tool` |
| auto JSON schema from the signature | `Tool` builds the OpenAI schema |
| framework parses the LLM's tool-call | `parse_tool_calls` (Hermes **and** OpenAI) |
| framework invokes the function | `registry.dispatch(name, args)` |
| multi-turn function-call loop | `run_tool_loop(...)` |

**Migrating to real LiveKit is a transport swap, not a rewrite:** keep the same functions, register
them with `livekit.agents.function_tool`, and let LiveKit Cloud carry the WebRTC/SIP audio. The
tools, the directory, and the loop don't change.

## What you see in the demo
- **Type a name** (English, 中文, or unknown) or **speak** (a small CPU Whisper transcribes first).
- The **tool-calling trace**: the assistant's `tool_call` JSON → the `search_contacts` tool
  response (the ranked matches) → the assistant's spoken reply.
- The ranked candidate table + score bars show **how the DB is queried** and **who is located**.

Try: `蔡孟儒` (resolve), `Tseng` (surname only → clarify), `David Miller` (unknown → not found).

> On this **free CPU** Space the model's turn is *simulated* — the 3B audio model needs a GPU —
> but the **tool protocol and dispatch are real** (`tools.py` runs unchanged). Swapping in the
> GPU-served Qwen-Omni changes only *who emits the tool call*; the loop, tools, and DB are identical.
