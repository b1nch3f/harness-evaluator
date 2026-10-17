"""Scripted ADK models for offline tests (no network)."""

from typing import AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import PrivateAttr


def call(name, **args):
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


def text(t):
    return types.Part(text=t)


class Scripted(BaseLlm):
    """Replies from a script; records every request it receives."""

    model: str = "scripted"
    _script: list = PrivateAttr(default_factory=list)
    requests: list = []

    def __init__(self, script):
        super().__init__(model="scripted")
        self._script = list(script)
        self.requests = []

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        self.requests.append(llm_request)
        parts = self._script.pop(0)
        yield LlmResponse(content=types.Content(role="model", parts=parts),
                          usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=10, candidates_token_count=2))
