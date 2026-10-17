"""Agent model: gpt-oss-120b behind an OpenAI-compatible endpoint, via ADK's LiteLlm."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from skills_builder.config import env

REASONING_EFFORTS = ("low", "medium", "high")
REQUEST_TIMEOUT_S = 600


@dataclass(frozen=True)
class ModelSettings:
    model: str
    api_base: str
    api_key: str
    temperature: float
    reasoning_effort: str

    def public(self) -> dict:
        """Settings safe to log or commit (no key; api_base is reduced to a label)."""
        d = asdict(self)
        d.pop("api_key")
        d["api_base"] = endpoint_label(self.api_base)
        return d


def endpoint_label(api_base: str) -> str:
    """A short, credential-free name for an endpoint, for result configs."""
    from urllib.parse import urlparse

    u = urlparse(api_base)
    return f"{u.hostname}{u.path}".rstrip("/") if u.hostname else "unknown"


def _require(name: str) -> str:
    value = env(name)
    if not value:
        raise RuntimeError(f"{name} is not set in .env")
    return value


def agent_settings() -> ModelSettings:
    effort = env("AGENT_REASONING_EFFORT", "medium")
    if effort not in REASONING_EFFORTS:
        raise RuntimeError(f"AGENT_REASONING_EFFORT must be one of {REASONING_EFFORTS}")
    return ModelSettings(
        model=_require("AGENT_MODEL"),
        api_base=_require("AGENT_API_BASE"),
        api_key=_require("AGENT_API_KEY"),
        temperature=float(env("AGENT_TEMPERATURE", "0.0")),
        reasoning_effort=effort,
    )


def completion_kwargs(s: ModelSettings) -> dict:
    """Arguments for litellm.completion.

    vLLM takes reasoning_effort as a top-level request field. LiteLLM rejects it
    for generic OpenAI-compatible models, so it goes through extra_body, which
    LiteLLM merges into the request body unchanged.
    """
    return {
        "api_base": s.api_base,
        "api_key": s.api_key,
        "temperature": s.temperature,
        "extra_body": {"reasoning_effort": s.reasoning_effort},
        "timeout": REQUEST_TIMEOUT_S,
    }


def build_agent_model(s: ModelSettings | None = None):
    from google.adk.models.lite_llm import LiteLlm

    s = s or agent_settings()
    return LiteLlm(model=s.model, **completion_kwargs(s))
