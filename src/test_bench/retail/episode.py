"""One retail conversation: ADK agent <-> ADK simulated user, then scoring.

Everything here is our own code on ADK (CLAUDE.md rule 2). Behavior follows
tau2's half-duplex orchestrator: the agent greets first; each turn is either
text or tool calls; tool calls run against this episode's database; the
episode ends on a user stop token, max steps, too many tool errors, or an
empty message.
"""

from __future__ import annotations

import asyncio
import functools
import json
import time
import uuid
from contextlib import aclosing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.adk.tools.function_tool import FunctionTool
from google.genai import types

from test_bench import data
from test_bench.model import ModelSettings, agent_settings, build_model, judge_settings, user_settings
from test_bench.harness import Harness
from test_bench.prompts import user_prompt
from test_bench.retail.models import RetailDB
from test_bench.retail.runtime import result_text, to_plain
from test_bench.retail.scorer import JudgeError, score_episode
from test_bench.retail.tools import TOOL_NAMES, RetailTools

GREETING = "Hi! How can I help you today?"
STOP_TOKENS = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")
MAX_TOOL_ERRORS = 10
AGENT_NAME = "retail_agent"
USER_NAME = "customer"
JUDGE_NAME = "judge"
APP_NAME = "test_bench"


class InfraError(RuntimeError):
    """Transport or parsing failure outside the model's control; the episode can be retried."""


# --- ADK plumbing ------------------------------------------------------------


def strip_identity(agent_name: str):
    """A before_model_callback that removes ADK's automatic identity line.

    ADK appends 'You are an agent. Your internal name is "<name>".' (agents here
    have no description) as its own "\n\n"-separated piece; dropping that piece
    leaves the system prompt exactly as written in our prompt file.
    """
    line = f'You are an agent. Your internal name is "{agent_name}".'

    def callback(callback_context, llm_request):
        config = llm_request.config
        si = config.system_instruction if config else None
        if isinstance(si, str) and line in si:
            config.system_instruction = "\n\n".join(p for p in si.split("\n\n") if p != line)
        return None

    return callback


def _event_text(event: Event) -> str:
    if not event.content or not event.content.parts:
        return ""
    return "".join(p.text for p in event.content.parts if p.text and not p.thought)


async def _run_turn(runner: InMemoryRunner, session_id: str, text: str, abort: asyncio.Event | None = None) -> str:
    """Send one message to an ADK agent and return its final text reply."""
    final = ""
    async with aclosing(
        runner.run_async(
            user_id="u",
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=text)]),
            abort_signal=abort,
        )
    ) as events:
        async for event in events:
            if event.error_code:
                raise InfraError(f"{event.author}: {event.error_code}: {event.error_message}")
            if event.author != "user" and event.is_final_response() and not event.get_function_calls():
                final = _event_text(event)
            if abort is not None and abort.is_set():
                break
    return final


# --- episode state -----------------------------------------------------------


@dataclass
class EpisodeLog:
    task_id: str
    harness: str
    trial: int = 0
    reward: float = 0.0
    termination_reason: str = ""
    score: dict | None = None
    steps: int = 0
    tool_errors: int = 0
    agent_turns: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    skill_tool_calls: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)  # tau2-style trajectory
    usage: dict = field(default_factory=lambda: {"agent": {}, "user": {}, "judge": {}})
    seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class _Episode:
    def __init__(self, task: dict, max_steps: int, log: EpisodeLog):
        self.task = task
        self.max_steps = max_steps
        self.log = log
        self.tools = RetailTools(RetailDB.fresh())
        self.ended: str | None = None  # termination reason once set
        self.abort = asyncio.Event()

    def end(self, reason: str) -> None:
        if self.ended is None:
            self.ended = reason
            self.abort.set()

    def add_step(self, n: int = 1) -> None:
        self.log.steps += n
        if self.log.steps >= self.max_steps:
            self.end("max_steps")

    # One FunctionTool per retail tool, running against this episode's database.
    def adk_tools(self) -> list[FunctionTool]:
        return [FunctionTool(self._wrap(name)) for name in TOOL_NAMES]

    def _wrap(self, name: str):
        method = getattr(self.tools, name)

        @functools.wraps(method)
        def tool(*args, **kwargs):
            return self._execute(name, method, kwargs)

        return tool

    def _execute(self, name: str, method, kwargs: dict) -> Any:
        if self.ended:
            return {"error": "The conversation has ended."}
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        try:
            result = method(**kwargs)
            content, error = result_text(result), False
            response = to_plain(result)
            if not isinstance(response, dict):
                response = {"result": response}
        except Exception as e:
            content, error = f"Error: {e}", True
            response = {"error": str(e)}
        self.log.messages.append(
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": call_id, "name": name, "arguments": kwargs}]}
        )
        self.log.messages.append({"role": "tool", "id": call_id, "content": content, "error": error})
        self.log.tool_calls.append({"id": call_id, "name": name, "arguments": kwargs, "result": content, "error": error})
        self.add_step(2)  # tool call + tool result, as in tau2's orchestrator
        if error:
            self.log.tool_errors += 1
            if self.log.tool_errors >= MAX_TOOL_ERRORS:
                self.end("too_many_errors")
        return response

    def after_tool(self, tool, args, tool_context, tool_response=None, **_):
        """Record skill-tool calls (they run locally in ADK and never touch the database)."""
        if tool.name not in TOOL_NAMES:
            self.log.skill_tool_calls.append(
                {"name": tool.name, "arguments": dict(args or {}), "response": _short(tool_response),
                 "at_message": len(self.log.messages)}
            )
        return None


def _short(obj: Any, limit: int = 4000) -> Any:
    text = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _usage_recorder(bucket: dict):
    def after_model(callback_context, llm_response):
        u = llm_response.usage_metadata
        if u is not None:
            bucket["calls"] = bucket.get("calls", 0) + 1
            bucket["prompt_tokens"] = bucket.get("prompt_tokens", 0) + (u.prompt_token_count or 0)
            bucket["completion_tokens"] = bucket.get("completion_tokens", 0) + (u.candidates_token_count or 0)
        return None

    return after_model


# --- running -----------------------------------------------------------------


@dataclass
class EpisodeSpec:
    task_id: str
    harness: Harness
    trial: int = 0
    max_steps: int = 100
    agent: ModelSettings | None = None
    user: ModelSettings | None = None
    judge: ModelSettings | None = None
    # Tests inject models (anything ADK's LlmAgent accepts as `model`).
    agent_model: Any = None
    user_model: Any = None
    judge_model: Any = None


def _skill_toolset(skills_dir: Path):
    from google.adk.skills import load_skills_from_dir
    from google.adk.tools.skill_toolset import SkillToolset

    if not skills_dir.is_dir():
        raise FileNotFoundError(f"skills dir not found: {skills_dir}")
    return SkillToolset(skills=load_skills_from_dir(skills_dir))


async def run_episode_async(spec: EpisodeSpec) -> EpisodeLog:
    task = data.tasks_by_id()[spec.task_id]
    exp = spec.harness
    if exp.skills_dir is not None and not exp.skill_names:
        raise ValueError(f"harness {exp.name!r} has a skills folder with no skills in it yet")

    log = EpisodeLog(task_id=spec.task_id, harness=exp.name, trial=spec.trial)
    ep = _Episode(task, spec.max_steps, log)
    t0 = time.time()

    agent_model = spec.agent_model or build_model(spec.agent or agent_settings())
    user_model = spec.user_model or build_model(spec.user or user_settings())

    tools: list = ep.adk_tools()
    if exp.skills_dir is not None:
        tools.append(_skill_toolset(exp.skills_dir))
    agent = LlmAgent(
        name=AGENT_NAME,
        model=agent_model,
        instruction=lambda _ctx: exp.prompt_text,  # callable: no {placeholder} injection
        tools=tools,
        before_model_callback=strip_identity(AGENT_NAME),
        after_model_callback=_usage_recorder(log.usage["agent"]),
        after_tool_callback=ep.after_tool,
    )
    user_text_prompt = user_prompt(task)
    user = LlmAgent(
        name=USER_NAME,
        model=user_model,
        instruction=lambda _ctx: user_text_prompt,
        before_model_callback=strip_identity(USER_NAME),
        after_model_callback=_usage_recorder(log.usage["user"]),
    )
    agent_runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
    user_runner = InMemoryRunner(agent=user, app_name=APP_NAME)
    agent_session = await agent_runner.session_service.create_session(app_name=APP_NAME, user_id="u")
    user_session = await user_runner.session_service.create_session(app_name=APP_NAME, user_id="u")

    # The agent opens with tau2's fixed greeting.
    await agent_runner.session_service.append_event(
        agent_session,
        Event(invocation_id="greeting", author=AGENT_NAME,
              content=types.Content(role="model", parts=[types.Part(text=GREETING)])),
    )
    log.messages.append({"role": "assistant", "content": GREETING})
    agent_text = GREETING

    try:
        while ep.ended is None:
            user_text = await _run_turn(user_runner, user_session.id, agent_text)
            if not user_text.strip():
                ep.end("user_error")
                break
            log.messages.append({"role": "user", "content": user_text})
            if any(tok in user_text for tok in STOP_TOKENS):
                ep.end("user_stop")
                break
            ep.add_step()
            if ep.ended:
                break

            agent_text = await _run_turn(agent_runner, agent_session.id, user_text, ep.abort)
            if ep.ended:
                break
            log.agent_turns += 1
            if not agent_text.strip():
                ep.end("agent_error")  # tau2 ends the episode on an empty agent message
                break
            log.messages.append({"role": "assistant", "content": agent_text})
            ep.add_step()

        log.termination_reason = ep.ended or "unknown"
        judge = _judge_fn(spec, log)
        try:
            score = await score_episode(task, log.termination_reason, ep.tools.db, log.messages, judge)
        except JudgeError as e:
            raise InfraError(str(e)) from e
        log.score = score.to_dict()
        log.reward = score.reward
    except Exception as e:
        log.error = f"{type(e).__name__}: {e}"
        raise
    finally:
        log.seconds = round(time.time() - t0, 2)
    return log


def _judge_fn(spec: EpisodeSpec, log: EpisodeLog):
    async def judge(system_prompt: str, user_prompt_text: str) -> str:
        model = spec.judge_model or build_model(spec.judge or judge_settings())
        agent = LlmAgent(
            name=JUDGE_NAME,
            model=model,
            instruction=lambda _ctx: system_prompt,
            before_model_callback=strip_identity(JUDGE_NAME),
            after_model_callback=_usage_recorder(log.usage["judge"]),
        )
        runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
        session = await runner.session_service.create_session(app_name=APP_NAME, user_id="u")
        return await _run_turn(runner, session.id, user_prompt_text)

    return judge


def run_episode(spec: EpisodeSpec) -> EpisodeLog:
    """Run one episode in its own event loop (safe to call from a worker thread)."""
    return asyncio.run(run_episode_async(spec))


def default_spec(task_id: str, harness: Harness, trial: int = 0, agent_profile: str = "AGENT",
                 max_steps: int = 100) -> EpisodeSpec:
    """A spec with the agent from a .env profile and the user and judge from .env."""
    return EpisodeSpec(task_id=task_id, harness=harness, trial=trial, max_steps=max_steps,
                       agent=agent_settings(agent_profile))
