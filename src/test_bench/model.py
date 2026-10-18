"""Models for the three roles (agent, simulated user, judge), via ADK's LiteLlm."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from test_bench.config import env

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
    top_p: float | None = None  # None = provider default
    max_tokens: int | None = None  # output cap per reply; None = provider default
    timeout: int = REQUEST_TIMEOUT_S  # seconds per request

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
    if effort == "none":  # for models that take no reasoning_effort
        return None
    if effort is not None and effort not in REASONING_EFFORTS:
        raise RuntimeError(f"{name} must be one of {REASONING_EFFORTS}")
    return effort


def agent_settings(profile: str = "AGENT", model: str | None = None, temperature: float | None = None,
                   reasoning_effort: str | None = None) -> ModelSettings:
    """Agent model settings from .env, read under a profile prefix.

    The default profile reads AGENT_MODEL, AGENT_API_BASE, AGENT_API_KEY,
    AGENT_TEMPERATURE and AGENT_REASONING_EFFORT. Another LLM is another
    prefix, for example QWEN_MODEL, QWEN_API_BASE, QWEN_API_KEY. Arguments
    given here (the `sb eval` flags) override the .env values.
    """
    p = profile.upper()
    effort = _effort(f"{p}_REASONING_EFFORT", "medium")
    if reasoning_effort is not None:
        if reasoning_effort not in (*REASONING_EFFORTS, "none"):
            raise RuntimeError(f"reasoning_effort must be one of {REASONING_EFFORTS} or none")
        effort = None if reasoning_effort == "none" else reasoning_effort
    return ModelSettings(
        model=model or _require(f"{p}_MODEL"),
        api_base=_require(f"{p}_API_BASE"),
        api_key=_require(f"{p}_API_KEY"),
        temperature=float(temperature if temperature is not None else env(f"{p}_TEMPERATURE", "0.0")),
        reasoning_effort=effort,
        **_sampling(p),
    )


def _sampling(prefix: str) -> dict:
    """Optional <PREFIX>_TOP_P, <PREFIX>_MAX_TOKENS and <PREFIX>_TIMEOUT from .env."""
    top_p, max_tokens, timeout = env(f"{prefix}_TOP_P"), env(f"{prefix}_MAX_TOKENS"), env(f"{prefix}_TIMEOUT")
    return {
        "top_p": float(top_p) if top_p else None,
        "max_tokens": int(max_tokens) if max_tokens else None,
        "timeout": int(timeout) if timeout else REQUEST_TIMEOUT_S,
    }


def user_settings() -> ModelSettings:
    return ModelSettings(
        model=_require("USER_MODEL"),
        api_base=env("USER_API_BASE") or _require("AGENT_API_BASE"),
        api_key=env("USER_API_KEY") or _require("AGENT_API_KEY"),
        temperature=float(env("USER_TEMPERATURE", "0.0")),
        reasoning_effort=_effort("USER_REASONING_EFFORT", None),
        **_sampling("USER"),
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
        **{k: v for k, v in _sampling("JUDGE").items() if v not in (None, REQUEST_TIMEOUT_S)},
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
        "timeout": s.timeout,
        "num_retries": NUM_RETRIES,
    }
    if s.top_p is not None:
        kw["top_p"] = s.top_p
    if s.max_tokens is not None:
        kw["max_tokens"] = s.max_tokens
    if s.reasoning_effort:
        kw["extra_body"] = {"reasoning_effort": s.reasoning_effort}
    return kw


def build_model(s: ModelSettings):
    from google.adk.models.lite_llm import LiteLlm

    return LiteLlm(model=s.model, **completion_kwargs(s))


def build_agent_model(s: ModelSettings | None = None):
    return build_model(s or agent_settings())
