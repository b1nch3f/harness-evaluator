"""Models for the three roles (agent, simulated user, judge), via ADK's LiteLlm."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from skills_builder.config import env

REASONING_EFFORTS = ("low", "medium", "high")
REQUEST_TIMEOUT_S = 600
NUM_RETRIES = 3  # transport retries inside litellm (tau2 uses 3)


@dataclass(frozen=True)
class ModelSettings:
    model: str
    api_base: str
    api_key: str
    temperature: float
    reasoning_effort: str | None

    def public(self) -> dict:
        """Settings safe to log or commit (no key; api_base reduced to a label)."""
        d = asdict(self)
        d.pop("api_key")
        d["api_base"] = endpoint_label(self.api_base)
        return d


def endpoint_label(api_base: str) -> str:
    """A short, credential-free name for an endpoint, for result configs."""
    from urllib.parse import urlparse

    u = urlparse(api_base or "")
    return f"{u.hostname}{u.path}".rstrip("/") if u.hostname else "unknown"


def _require(name: str) -> str:
    value = env(name)
    if not value:
        raise RuntimeError(f"{name} is not set in .env")
    return value


def _effort(name: str, default: str | None) -> str | None:
    effort = env(name, default)
    if effort is not None and effort not in REASONING_EFFORTS:
        raise RuntimeError(f"{name} must be one of {REASONING_EFFORTS}")
    return effort


def agent_settings() -> ModelSettings:
    return ModelSettings(
        model=_require("AGENT_MODEL"),
        api_base=_require("AGENT_API_BASE"),
        api_key=_require("AGENT_API_KEY"),
        temperature=float(env("AGENT_TEMPERATURE", "0.0")),
        reasoning_effort=_effort("AGENT_REASONING_EFFORT", "medium"),
    )


def user_settings() -> ModelSettings:
    return ModelSettings(
        model=_require("USER_MODEL"),
        api_base=env("USER_API_BASE") or _require("AGENT_API_BASE"),
        api_key=env("USER_API_KEY") or _require("AGENT_API_KEY"),
        temperature=float(env("USER_TEMPERATURE", "0.0")),
        reasoning_effort=_effort("USER_REASONING_EFFORT", None),
    )


def judge_settings() -> ModelSettings:
    """The judge defaults to the user-simulator settings; temperature is always 0."""
    user = user_settings()
    return replace(
        user,
        model=env("JUDGE_MODEL") or user.model,
        api_base=env("JUDGE_API_BASE") or user.api_base,
        api_key=env("JUDGE_API_KEY") or user.api_key,
        temperature=0.0,
        reasoning_effort=_effort("JUDGE_REASONING_EFFORT", None) or user.reasoning_effort,
    )


def completion_kwargs(s: ModelSettings) -> dict:
    """Arguments for litellm.completion.

    vLLM takes reasoning_effort as a request-body field. LiteLLM rejects it for
    generic OpenAI-compatible models, so it goes through extra_body.
    """
    kw = {
        "api_base": s.api_base,
        "api_key": s.api_key,
        "temperature": s.temperature,
        "timeout": REQUEST_TIMEOUT_S,
        "num_retries": NUM_RETRIES,
    }
    if s.reasoning_effort:
        kw["extra_body"] = {"reasoning_effort": s.reasoning_effort}
    return kw


def build_model(s: ModelSettings):
    from google.adk.models.lite_llm import LiteLlm

    return LiteLlm(model=s.model, **completion_kwargs(s))


def build_agent_model(s: ModelSettings | None = None):
    return build_model(s or agent_settings())
