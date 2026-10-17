"""End-to-end episode with scripted models (no network)."""

from pathlib import Path
from typing import AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import PrivateAttr

from skills_builder.prompts import agent_prompt, user_prompt
from skills_builder.retail import episode as ep
from skills_builder import data

FIXTURE_SKILLS = Path(__file__).parent / "fixtures" / "skills"
TASK_ID = "0"  # train: Yusuf Rossi, zip 19122, exchange in order #W2378156


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


def _spec(condition, agent, user, judge=None, skills_dir=None):
    return ep.EpisodeSpec(task_id=TASK_ID, condition=condition, skills_dir=skills_dir,
                          agent_model=agent, user_model=user, judge_model=judge or Scripted([]))


def test_tool_calls_change_the_episode_database_and_get_scored():
    agent = Scripted([
        [call("find_user_id_by_name_zip", first_name="Yusuf", last_name="Rossi", zip="19122")],
        [call("get_order_details", order_id="#W0000000")],  # does not exist -> tool error
        [text("I found your account. What would you like to exchange?")],
    ])
    user = Scripted([
        [text("Hi, I'm Yusuf Rossi, zip 19122. I want to exchange two items.")],
        [text("Never mind. ###STOP###")],
    ])
    log = ep.run_episode(_spec("A_baseline", agent, user))

    assert log.termination_reason == "user_stop"
    assert [c["name"] for c in log.tool_calls] == ["find_user_id_by_name_zip", "get_order_details"]
    assert log.tool_calls[0]["result"] == "yusuf_rossi_9620"
    assert log.tool_calls[1]["error"] is True and log.tool_calls[1]["result"] == "Error: Order not found"
    assert log.tool_errors == 1
    # No exchange happened, so the database check fails and the reward is 0.
    assert log.score["db_match"] is False and log.reward == 0.0
    roles = [m["role"] for m in log.messages]
    assert roles == ["assistant", "user", "assistant", "tool", "assistant", "tool", "assistant", "user"]
    assert log.messages[0]["content"] == ep.GREETING

    # The agent's system prompt is exactly the prompt file (no ADK identity line).
    first = agent.requests[0]
    assert first.config.system_instruction == agent_prompt("A_baseline").text
    assert sorted(first.tools_dict) == sorted(ep.TOOL_NAMES)
    # History starts with the greeting, then the user's message.
    assert [c.role for c in first.contents] == ["model", "user"]
    assert first.contents[0].parts[0].text == ep.GREETING
    # The user simulator gets tau2's user prompt and sees the greeting as its first message.
    assert user.requests[0].config.system_instruction == user_prompt(data.tasks_by_id()[TASK_ID])
    assert user.requests[0].contents[-1].parts[0].text == ep.GREETING
    assert log.usage["agent"]["calls"] == 3 and log.usage["user"]["calls"] == 2


def test_expected_actions_score_one():
    """Executing the task's expected actions gives a database match."""
    task = data.tasks_by_id()[TASK_ID]
    script = [[call(a["name"], **a["arguments"])] for a in task["evaluation_criteria"]["actions"]]
    agent = Scripted(script + [[text("Your exchange has been requested.")]])
    user = Scripted([[text("I want to exchange items in order #W2378156.")], [text("Thanks! ###STOP###")]])
    log = ep.run_episode(_spec("A_baseline", agent, user))
    assert log.termination_reason == "user_stop"
    assert log.score["db_match"] is True
    assert task["evaluation_criteria"].get("nl_assertions") is None
    assert log.reward == 1.0


def test_judge_runs_for_tasks_with_assertions():
    task_id = next(t["id"] for t in data.tasks() if t["evaluation_criteria"].get("nl_assertions")
                   and t["id"] in data.splits()["train"])
    task = data.tasks_by_id()[task_id]
    script = [[call(a["name"], **a["arguments"])] for a in task["evaluation_criteria"]["actions"]]
    agent = Scripted(script + [[text("Done.")]])
    user = Scripted([[text("Hello.")], [text("###STOP###")]])
    assertions = task["evaluation_criteria"]["nl_assertions"]
    reply = '```json\n{"results": [' + ",".join(
        f'{{"expectedOutcome": {a!r}, "reasoning": "x", "metExpectation": false}}'.replace("'", '"') for a in assertions
    ) + "]}\n```"
    judge = Scripted([[text(reply)]])
    spec = ep.EpisodeSpec(task_id=task_id, condition="A_baseline", agent_model=agent, user_model=user, judge_model=judge)
    log = ep.run_episode(spec)
    assert log.score["db_match"] is True
    assert log.score["nl_reward"] == 0.0 and log.reward == 0.0
    sent = judge.requests[0]
    assert "expectedOutcomes" in sent.contents[-1].parts[0].text
    assert "assistant: " + ep.GREETING in sent.contents[-1].parts[0].text


def test_empty_agent_reply_is_an_agent_error():
    agent = Scripted([[text("")]])
    user = Scripted([[text("Hi.")]])
    log = ep.run_episode(_spec("A_baseline", agent, user))
    assert log.termination_reason == "agent_error" and log.reward == 0.0


def test_max_steps_ends_the_episode():
    agent = Scripted([[call("list_all_product_types")]] * 60)
    user = Scripted([[text("Hi.")]])
    spec = _spec("A_baseline", agent, user)
    spec.max_steps = 10
    log = ep.run_episode(spec)
    assert log.termination_reason == "max_steps" and log.reward == 0.0
    assert log.steps >= 10


def test_bare_condition_has_no_policy():
    agent = Scripted([[text("How can I help?")]])
    user = Scripted([[text("Hi.")], [text("###STOP###")]])
    ep.run_episode(_spec("A0_bare", agent, user))
    si = agent.requests[0].config.system_instruction
    assert si == agent_prompt("A0_bare").text and "</policy>" not in si


def test_skills_load_locally_and_never_touch_the_database():
    agent = Scripted([
        [call("list_skills")],
        [call("load_skill", skill_name="harness-check")],
        [text("How can I help with your exchange?")],
    ])
    user = Scripted([[text("Hi, I need an exchange.")], [text("###STOP###")]])
    log = ep.run_episode(_spec("B_skills", agent, user, skills_dir=FIXTURE_SKILLS))
    assert log.termination_reason == "user_stop"
    assert [c["name"] for c in log.skill_tool_calls] == ["list_skills", "load_skill"]
    assert "harness-check" in log.skill_tool_calls[1]["response"]
    assert log.tool_calls == []
    first = agent.requests[0]
    assert "list_skills" in first.tools_dict and "load_skill" in first.tools_dict
    # The policy prompt comes first; SkillToolset may append its own text after it.
    assert first.config.system_instruction.startswith(agent_prompt("B_skills").text)
