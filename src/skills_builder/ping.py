"""`sb ping`: one chat request and one tool-call request through ADK's LiteLlm."""

from __future__ import annotations

import asyncio
import time

from google.adk.models.llm_request import LlmRequest
from google.genai import types

from skills_builder.model import agent_settings, build_agent_model

PING_TOOL = types.FunctionDeclaration(
    name="get_order_details",
    description="Get the status and details of an order.",
    parameters_json_schema={
        "type": "object",
        "properties": {"order_id": {"type": "string", "description": "The order id, such as '#W0000000'."}},
        "required": ["order_id"],
    },
)


async def _generate(model, contents, tools=None):
    config = types.GenerateContentConfig(
        system_instruction="You are a retail customer service agent.",
        tools=[types.Tool(function_declarations=tools)] if tools else None,
    )
    req = LlmRequest(model=model.model, contents=contents, config=config)
    last = None
    async for resp in model.generate_content_async(req, stream=False):
        last = resp
    return last


def _parts(resp):
    return list(resp.content.parts) if resp and resp.content and resp.content.parts else []


async def ping() -> bool:
    settings = agent_settings()
    model = build_agent_model(settings)
    print(f"model={settings.model} endpoint={settings.public()['api_base']} "
          f"temperature={settings.temperature} reasoning_effort={settings.reasoning_effort}")

    t = time.time()
    resp = await _generate(model, [types.Content(role="user", parts=[types.Part(text="Say hello in five words.")])])
    text = " ".join(p.text for p in _parts(resp) if p.text and not p.thought)
    print(f"[chat] {time.time() - t:.1f}s reply: {text!r}")
    chat_ok = bool(text.strip())

    t = time.time()
    resp = await _generate(
        model,
        [types.Content(role="user", parts=[types.Part(text="What is the status of my order #W2378156?")])],
        tools=[PING_TOOL],
    )
    calls = [p.function_call for p in _parts(resp) if p.function_call]
    for c in calls:
        print(f"[tool] {time.time() - t:.1f}s parsed tool call: {c.name}({dict(c.args or {})})")
    tool_ok = len(calls) == 1 and calls[0].name == PING_TOOL.name and "order_id" in (calls[0].args or {})
    if not calls:
        print(f"[tool] {time.time() - t:.1f}s NO tool call. Parts: {_parts(resp)}")

    print("PING OK" if chat_ok and tool_ok else "PING FAILED")
    return chat_ok and tool_ok


def main() -> bool:
    return asyncio.run(ping())
