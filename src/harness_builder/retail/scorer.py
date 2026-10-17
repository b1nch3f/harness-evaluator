"""Scoring, following tau2's rules for retail (see docs/01-experiment-protocol.md).

reward = 0 unless the conversation ended normally (user or agent stop);
otherwise reward = database check x assertion check.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Awaitable, Callable

from harness_builder.retail.models import RetailDB
from harness_builder.retail.tools import RetailTools

SCORED_TERMINATIONS = ("user_stop", "agent_stop")

# tau2's judge prompt (tau2/evaluator/evaluator_nl_assertions.py), unchanged.
JUDGE_SYSTEM_PROMPT = """
        TASK
        - You will be given a list of expected outcomes and a conversation that was collected during a test case run.
        - The conversation is between an agent and a customer.
        - Your job is to evaluate whether the agent satisfies each of the expected outcomes.
        - Grade each expected outcome individually.

        FORMAT
        - Your response should be a JSON object with the following fields:
        - `reasoning`: a short explanation for your classification
        - `metExpectation`: `true` if the agent satisfies the expected outcomes, `false` otherwise
        - `expectedOutcome`: repeat the expectation from the input that you are grading
        
        Example response structure:
        {
            "results": [
                {
                    "expectedOutcome": "<one of the expected outcomes from the input>",
                    "reasoning": "<reasoning trace>",
                    "metExpectation": <false or true>,
                }
            ]
        }
        """

JUDGE_USER_PROMPT = """
        conversation:
        {trajectory_str}
        
        expectedOutcomes:
        {nl_assertions}
        """


@dataclass
class Score:
    reward: float
    termination_reason: str
    db_match: bool | None = None
    nl_assertions: list[dict] = field(default_factory=list)
    nl_reward: float | None = None
    gold_action_errors: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class JudgeError(RuntimeError):
    """The judge's reply could not be parsed (an infrastructure error, not a model failure)."""


def gold_database(task: dict) -> tuple[RetailDB, list[str]]:
    """Fresh database with the task's expected actions applied (errors skipped, as tau2 does)."""
    tools = RetailTools(RetailDB.fresh())
    errors = []
    for action in task["evaluation_criteria"].get("actions") or []:
        try:
            tools.call(action["name"], dict(action["arguments"]))
        except Exception as e:
            errors.append(f"{action['name']}: {e}")
    return tools.db, errors


def db_check(task: dict, final_db: RetailDB) -> tuple[bool, list[str]]:
    gold, errors = gold_database(task)
    return gold.canonical() == final_db.canonical(), errors


def judge_transcript(messages: list[dict]) -> str:
    """tau2 renders every message, tool messages included, as 'role: content'."""
    return "\n".join(f"{m['role']}: {m.get('content')}" for m in messages)


def parse_judge_reply(text: str) -> list[dict]:
    """The judge's JSON. Tolerates a ```json fence around it."""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except ValueError as e:
        raise JudgeError(f"judge reply is not JSON: {text[:200]!r}") from e
    results = data.get("results", []) if isinstance(data, dict) else None
    if not isinstance(results, list):
        raise JudgeError("judge reply has no 'results' list")
    return [
        {
            "nl_assertion": r.get("expectedOutcome"),
            "met": bool(r.get("metExpectation")),
            "justification": r.get("reasoning"),
        }
        for r in results
    ]


Judge = Callable[[str, str], Awaitable[str]]  # (system_prompt, user_prompt) -> reply text


async def score_episode(
    task: dict,
    termination_reason: str,
    final_db: RetailDB,
    messages: list[dict],
    judge: Judge | None,
    judge_retries: int = 3,
) -> Score:
    if termination_reason not in SCORED_TERMINATIONS:
        return Score(reward=0.0, termination_reason=termination_reason,
                     note=f"ended early: {termination_reason}")

    criteria = task["evaluation_criteria"]
    match, gold_errors = db_check(task, final_db)
    reward = 1.0 if match else 0.0
    score = Score(reward=reward, termination_reason=termination_reason, db_match=match,
                  gold_action_errors=gold_errors)

    if "NL_ASSERTION" in (criteria.get("reward_basis") or []):
        assertions = criteria.get("nl_assertions") or []
        if not assertions:
            score.nl_reward = 1.0
        else:
            if judge is None:
                raise ValueError("task needs the judge but none was given")
            user_prompt = JUDGE_USER_PROMPT.format(
                trajectory_str=judge_transcript(messages), nl_assertions=assertions
            )
            last_error = None
            for _ in range(judge_retries):
                try:
                    score.nl_assertions = parse_judge_reply(await judge(JUDGE_SYSTEM_PROMPT, user_prompt))
                    break
                except JudgeError as e:
                    last_error = e
            else:
                raise last_error
            score.nl_reward = 1.0 if all(r["met"] for r in score.nl_assertions) else 0.0
        score.reward *= score.nl_reward
    return score
